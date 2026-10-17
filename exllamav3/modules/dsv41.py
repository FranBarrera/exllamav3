"""DeepSeek-V4.1 modules: CSA2 attention, delayed mHC, Engram, DSpark attention.

The forward follows the official MIT reference (inference/model.py). Shared compressed KV,
indexer top-k and the delayed pre-mix live in a bag on ``quant_preserve`` during conversion
and on ``DSV41State`` during generation, because calibration forwards one top-level module
at a time. Writes are addressed by absolute position and ``start_pos == 0`` rebuilds partial
compressor state, so the capture and reference passes (both at position 0) agree.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from typing_extensions import override

from ..architecture import dsv41_math as M
from ..model.config import Config
from . import Module, Linear, RMSNorm, BlockSparseMLP
from .hyperconnections import HyperConnection


def start_position(params: dict) -> int:
    states = params.get("recurrent_states")
    if states:
        return int(states[0].position)
    if "past_len" in params:
        return int(params["past_len"])
    position = params.get("position")
    if isinstance(position, int):
        return position
    return 0


class _Runtime:
    """One bag per sequence. Conversion uses a single bag and keeps it on the CPU."""

    def __init__(self, bags: list[dict], to_cpu: bool):
        self.bags = bags
        self.to_cpu = to_cpu

    def get(self, key: str, device=None) -> torch.Tensor | None:
        parts = []
        for bag in self.bags:
            value = bag.get(key)
            if value is None:
                return None
            parts.append(value)
        value = parts[0] if len(parts) == 1 else torch.cat(parts, dim = 0)
        if device is not None and value.device != device:
            value = value.to(device)
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if len(self.bags) == 1:
            stored = value.detach()
            self.bags[0][key] = stored.cpu() if self.to_cpu else stored
            return
        for index, bag in enumerate(self.bags):
            stored = value[index:index + 1].detach()
            bag[key] = stored.cpu() if self.to_cpu else stored


def bind_runtime(params: dict, batch: int) -> _Runtime:
    preserve = params.get("quant_preserve")
    if isinstance(preserve, dict):
        return _Runtime([preserve.setdefault("dsv41", {})], to_cpu = True)
    states = params.get("recurrent_states")
    if states:
        if len(states) != batch:
            raise RuntimeError(f"DeepSeek-V4.1 batch {batch} does not match {len(states)} recurrent states")
        return _Runtime([state.bag for state in states], to_cpu = False)
    return _Runtime([params.setdefault("dsv41", {})], to_cpu = False)


def _grow(buf: torch.Tensor | None, batch: int, rows: int, width: int, like: torch.Tensor, fill: float) -> torch.Tensor:
    if buf is not None and buf.shape[0] == batch and buf.shape[1] >= rows and buf.shape[-1] == width:
        return buf.clone()
    out = like.new_full((batch, max(rows, 1), width), fill)
    if buf is not None and buf.shape[0] == batch and buf.shape[-1] == width:
        n = min(buf.shape[1], out.shape[1])
        out[:, :n] = buf[:, :n].to(dtype = out.dtype)
    return out


class DSV41State:
    """Per-sequence bag for window KV, compressed KV and the Engram token cache.

    In-place rewind is not implemented (``guaranteed_rollback`` is 0). The generator
    restores a checkpoint and replays. Speculative decoding is not wired up.
    """

    exported = False
    guaranteed_rollback = 0

    def __init__(self, cache, slot: int, position: int, clear: bool = True, stashed: dict | None = None, test_state: bool = False):
        self.cache = cache
        self.slot = slot
        self.position = position
        self.last_history = 0
        self.bag: dict = {}
        self.checkpoint_size = 0
        if stashed is not None:
            self.unstash(stashed)
        elif clear:
            self.bag = {}

    def free(self):
        self.cache.release_state(self)

    def rewind(self, num_tokens: int):
        raise RuntimeError(
            "DeepSeek-V4.1 state has no in-place rewind; the generator restores a checkpoint instead"
        )

    def rollback_capacity(self):
        return 0

    def post_advance(self):
        pass

    def reset(self):
        self.position = 0
        self.last_history = 0
        self.bag = {}

    def stash(self) -> dict:
        bag = {}
        nbytes = 0
        for key, value in self.bag.items():
            if isinstance(value, torch.Tensor):
                cloned = value.detach().cpu().contiguous().clone()
                bag[key] = cloned
                nbytes += cloned.nbytes
        self.checkpoint_size = nbytes
        return {"position": self.position, "bag": bag, "checkpoint_size": nbytes}

    def unstash(self, stashed: dict):
        bag = {}
        for key, value in stashed.get("bag", {}).items():
            bag[key] = value.detach().clone() if isinstance(value, torch.Tensor) else value
        self.bag = bag
        self.checkpoint_size = int(stashed.get("checkpoint_size", 0))


def _engram_tables(config):
    cached = getattr(config, "_dsv41_engram_tables", None)
    if cached is not None:
        return cached
    ids = tuple(config.engram_layer_ids)
    primes = M.engram_primes(ids, config.engram_max_ngram_size, config.engram_n_heads, config.engram_vocab_size)
    prime_t = torch.tensor(primes, dtype = torch.int64)
    cached = {
        "ids": ids,
        "primes": prime_t,
        "offsets": M.engram_offsets(primes),
        "multipliers": M.hash_multipliers(ids, config.engram_max_ngram_size, config.engram_compressed_vocab_size),
    }
    config._dsv41_engram_tables = cached
    return cached


def _token_map(config) -> tuple[torch.Tensor, int]:
    cached = getattr(config, "_dsv41_token_map", None)
    if cached is not None:
        return cached
    path = os.path.join(config.directory, "tokenizer.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"DeepSeek-V4.1 Engram needs {path} to build the compressed token map"
        )
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(path)
    lookup, size = M.build_compressed_token_map(tokenizer)
    if size != config.engram_compressed_vocab_size:
        raise RuntimeError(
            f"Engram compressed vocab is {size}, config says {config.engram_compressed_vocab_size}"
        )
    pad = lookup[config.engram_pad_id]
    cached = (torch.tensor(lookup, dtype = torch.int64), pad)
    config._dsv41_token_map = cached
    return cached


class DSV41Engram(Module):
    """N-gram hash lookup added to the residual before the block. The table stays on disk."""

    def __init__(self, config, key: str, layer_idx: int, hidden_size: int, hc_mult: int, qbits_key: str = "bits"):
        super().__init__(config, key, None)
        self.layer_idx = None  # top-level layer index belongs to the block
        self.engram_layer = layer_idx
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = config.rms_norm_eps
        layout = _engram_tables(config)
        self.hash_index = layout["ids"].index(layer_idx)
        self.n_heads = config.engram_n_heads
        self.head_dim = config.engram_head_dim
        self.max_ngram = config.engram_max_ngram_size
        self.n_cols = (self.max_ngram - 1) * self.n_heads
        self.wkv = Linear(
            config, f"{key}.wkv",
            self.n_cols * self.head_dim,
            hidden_size * (hc_mult + 1),
            qmap = f"{key}.wkv",
            out_dtype = torch.float,
            qbits_key = qbits_key,
            trim_padded_out = True,
        )
        self.register_submodule(self.wkv)
        self.q_weight = None
        self.k_weight = None
        self.embed_weight = None
        self.embed_scale = None
        self.extra_source_keys = (
            f"{key}.q_weight",
            f"{key}.k_weight",
            f"{key}.embed.weight",
            f"{key}.embed.scale",
        )

    @override
    def optimizer_targets(self):
        return [self.wkv.key]

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        stc = self.config.stc
        self.q_weight = stc.get_tensor(f"{self.key}.q_weight", device, allow_bf16 = True).float()
        self.k_weight = stc.get_tensor(f"{self.key}.k_weight", device, allow_bf16 = True).float()
        self.embed_weight = stc.get_tensor_handle(f"{self.key}.embed.weight")
        self.embed_scale = stc.get_tensor_handle(f"{self.key}.embed.scale")

    @override
    def unload(self):
        super().unload()
        for handle in (self.embed_weight, self.embed_scale):
            if handle is not None:
                handle.close()
        self.q_weight = self.k_weight = None
        self.embed_weight = self.embed_scale = None

    @override
    def get_tensors(self):
        return {
            f"{self.key}.q_weight": self.q_weight.contiguous(),
            f"{self.key}.k_weight": self.k_weight.contiguous(),
        }

    def _hashes(self, input_ids: torch.Tensor, start: int, runtime: _Runtime) -> torch.Tensor:
        token_map, pad_id = _token_map(self.config)
        token_map = token_map.to(input_ids.device)
        compressed = token_map[input_ids]
        cache = runtime.get("engram_cache", device = input_ids.device)
        need = start + input_ids.shape[1]
        if cache is None or cache.shape[0] != input_ids.shape[0] or cache.shape[1] < need:
            grown = input_ids.new_zeros(input_ids.shape[0], max(need, 1))
            if cache is not None and cache.shape[0] == input_ids.shape[0]:
                grown[:, :cache.shape[1]] = cache[:, :grown.shape[1]]
            cache = grown
        tables = _engram_tables(self.config)
        index = self.hash_index
        hashes = M.ngram_hash_ids(
            compressed,
            cache,
            start,
            tables["primes"][index:index + 1],
            tables["offsets"][index:index + 1],
            tables["multipliers"][index:index + 1],
            pad_id,
        )
        runtime.put("engram_cache", cache)
        return hashes[:, :, 0, :]

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        runtime = bind_runtime(params, x.shape[0])
        start = start_position(params)
        input_ids = params["input_ids"]
        if input_ids.shape[0] != x.shape[0] or input_ids.shape[1] != x.shape[1]:
            raise RuntimeError(
                f"Engram input_ids {tuple(input_ids.shape)} does not match hidden {tuple(x.shape[:2])}"
            )
        hashes = self._hashes(input_ids, start, runtime)
        flat = hashes.reshape(-1).to(torch.long).cpu()
        weight = self.embed_weight.read_rows(flat)
        scale = self.embed_scale.read_rows(flat)
        rows = M.dequant_fp8_e8m0(weight, scale, 32).to(device = x.device, dtype = torch.float)
        rows = rows.view(*hashes.shape, self.head_dim).flatten(-2)
        kv = self.wkv.forward(rows.to(dtype = torch.half), params)
        key, value = kv.split([self.hc_mult * self.hidden_size, self.hidden_size], dim = -1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        weight = self.q_weight * self.k_weight
        gate = M.engram_gate(x, key, weight, self.eps)
        types = params.get("token_types")
        if types is not None:
            gate = gate.masked_fill(types.unsqueeze(-1) >= 0, 0)
        return (x.float() + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(dtype = x.dtype)


class DSV41Attention(Module):
    """Latent attention over a sliding window plus, when compress_ratio > 0, shared compressed KV."""

    def __init__(
        self,
        config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        num_q_heads: int,
        head_dim: int,
        rope_dim: int,
        q_lora_rank: int,
        o_groups: int,
        o_lora_rank: int,
        window: int,
        compress_ratio: int,
        is_kv_source: bool,
        is_index_source: bool,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        candidate_source: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        rope_theta: float,
        compress_rope_theta: float,
        yarn: dict,
        rms_norm_eps: float,
        qmap: str,
        qbits_key: str = "bits",
        select_hq_bits: int = 2,
        dspark: bool = False,
    ):
        super().__init__(config, key, qmap)
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.n_heads = num_q_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.o_groups = o_groups
        self.o_lora_rank = o_lora_rank
        self.window = window
        self.compress_ratio = compress_ratio
        self.is_kv_source = is_kv_source
        self.is_index_source = is_index_source
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source = candidate_source
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.is_candidate_source = layer_idx == candidate_source
        self.uses_candidates = 0 <= candidate_source < layer_idx
        self.dspark = dspark
        self.rms_norm_eps = rms_norm_eps
        self.q_priority = 2 + select_hq_bits
        self.extra_source_keys = (f"{key}.attn_sink",)
        self.attn_sink = None

        if compress_ratio:
            original = int(yarn.get("original_seq_len", 0))
            theta = compress_rope_theta
        else:
            original = 0
            theta = rope_theta
        self.inv_freq = M.yarn_inv_freq(
            rope_dim, theta, float(yarn.get("factor", 1.0)), original,
            float(yarn.get("beta_fast", 32)), float(yarn.get("beta_slow", 1)),
        )

        def linear(name, inp, out, qmap_, hq = select_hq_bits, pad = 128):
            return Linear(
                config, f"{key}.{name}", inp, out,
                qmap = qmap_, out_dtype = torch.half, qbits_key = qbits_key,
                trim_padded_out = True, select_hq_bits = hq, pad_to = pad,
            )

        self.q_a = linear("wq_a", hidden_size, q_lora_rank, qmap)
        self.q_norm = RMSNorm(config, f"{key}.q_norm", rms_norm_eps)
        self.q_b = linear("wq_b", q_lora_rank, num_q_heads * head_dim, f"{key}.q_b")
        self.wkv = linear("wkv", hidden_size, head_dim, qmap)
        self.kv_norm = RMSNorm(config, f"{key}.kv_norm", rms_norm_eps)
        group_in = num_q_heads * head_dim // o_groups
        self.wo_a = [
            Linear(
                config, f"{key}.wo_a.slice.{g}", group_in, o_lora_rank,
                fkey = f"{key}.wo_a",
                frange = [g * o_lora_rank, (g + 1) * o_lora_rank],
                frange_dim = 0,
                qmap = f"{key}.o.{g}",
                out_dtype = torch.half,
                qbits_key = qbits_key,
                trim_padded_out = True,
                select_hq_bits = select_hq_bits,
            )
            for g in range(o_groups)
        ]
        self.wo_b = Linear(
            config, f"{key}.wo_b", o_groups * o_lora_rank, hidden_size,
            qmap = f"{key}.o_b", out_dtype = torch.float, qbits_key = qbits_key,
            trim_padded_out = True, select_hq_bits = select_hq_bits,
        )
        self.compressor_wkv = None
        self.compressor_wgate = None
        self.compressor_norm = None
        if is_kv_source:
            comp_dtype_qmap = qmap
            self.compressor_wkv = linear("compressor.wkv", hidden_size, head_dim, comp_dtype_qmap)
            self.compressor_norm = RMSNorm(config, f"{key}.compressor.norm", rms_norm_eps)
            if compress_ratio > 1:
                self.compressor_wgate = linear("compressor.wgate", hidden_size, head_dim, comp_dtype_qmap)
        self.idx_wq_b = None
        self.idx_weights = None
        self.idx_wk = None
        self.idx_k_norm = None
        if is_index_source:
            self.idx_wq_b = linear(
                "indexer.wq_b", q_lora_rank, index_n_heads * index_head_dim, f"{key}.idx.q",
            )
            self.idx_weights = Linear(
                config, f"{key}.indexer.weights_proj", hidden_size, index_n_heads,
                qmap = None, out_dtype = torch.half, pad_to = 1,
            )
            if is_kv_source:
                self.idx_wk = linear("indexer.wk", head_dim, index_head_dim, qmap)
                self.idx_k_norm = RMSNorm(config, f"{key}.indexer.k_norm", rms_norm_eps)
        for module in (
            self.q_a, self.q_norm, self.q_b, self.wkv, self.kv_norm, *self.wo_a, self.wo_b,
            self.compressor_wkv, self.compressor_wgate, self.compressor_norm,
            self.idx_wq_b, self.idx_weights, self.idx_wk, self.idx_k_norm,
        ):
            self.register_submodule(module)

    @override
    def optimizer_targets(self):
        return [module.key for module in self if isinstance(module, Linear) and module.qmap]

    @override
    def load(self, device, **kwargs):
        super().load(device, **kwargs)
        sink = self.config.stc.get_tensor(f"{self.key}.attn_sink", device, allow_bf16 = True)
        self.attn_sink = sink.float().contiguous()

    @override
    def unload(self):
        super().unload()
        self.attn_sink = None

    @override
    def get_tensors(self):
        if self.attn_sink is None:
            return {}
        return {f"{self.key}.attn_sink": self.attn_sink.contiguous()}

    def _freqs(self, positions: torch.Tensor) -> torch.Tensor:
        return M.rope_cis_at(positions, self.inv_freq.to(positions.device))

    def _window(self, runtime: _Runtime, kv: torch.Tensor, start: int):
        batch, seqlen, _ = kv.shape
        key = f"win.{self.layer_idx}"
        cache = _grow(runtime.get(key, kv.device), batch, self.window, kv.shape[-1], kv, 0.0)
        if start == 0:
            if seqlen <= self.window:
                cache[:, :seqlen] = kv
            else:
                cutoff = seqlen % self.window
                tail = kv[:, -self.window:]
                cache[:, cutoff:] = tail[:, :self.window - cutoff]
                cache[:, :cutoff] = tail[:, self.window - cutoff:]
            attended = kv
        else:
            cache[:, start % self.window] = kv.squeeze(1)
            attended = cache
        runtime.put(key, cache)
        idxs = M.window_topk_idxs(self.window, batch, seqlen, start, device = kv.device)
        return attended, idxs

    def _latent(self, x: torch.Tensor, params: dict, runtime: _Runtime, start: int) -> torch.Tensor | None:
        ratio = self.compress_ratio
        projected = self.compressor_wkv.forward(x, params)
        if ratio == 1:
            return self.compressor_norm.forward(projected, params)
        score = self.compressor_wgate.forward(x, params).float()
        projected_f = projected.float()
        batch, seqlen, width = projected_f.shape
        state_k = f"pkv.{self.layer_idx}"
        state_s = f"psc.{self.layer_idx}"
        if start == 0:
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            kv_state = projected_f.new_zeros(batch, ratio, width)
            score_state = projected_f.new_full((batch, ratio, width), float("-inf"))
            if remainder:
                kv_state[:, :remainder] = projected_f[:, cutoff:]
                score_state[:, :remainder] = score[:, cutoff:]
            runtime.put(state_k, kv_state)
            runtime.put(state_s, score_state)
            if cutoff == 0:
                return None
            grouped_k = projected_f[:, :cutoff].unflatten(1, (-1, ratio))
            grouped_s = score[:, :cutoff].unflatten(1, (-1, ratio))
            pooled = M.pool_groups(grouped_k, grouped_s)
        else:
            kv_state = runtime.get(state_k, x.device)
            score_state = runtime.get(state_s, x.device)
            if kv_state is None:
                kv_state = projected_f.new_zeros(batch, ratio, width)
                score_state = projected_f.new_full((batch, ratio, width), float("-inf"))
            else:
                kv_state = kv_state.clone()
                score_state = score_state.clone()
            slot = start % ratio
            kv_state[:, slot] = projected_f.squeeze(1)
            score_state[:, slot] = score.squeeze(1)
            runtime.put(state_k, kv_state)
            runtime.put(state_s, score_state)
            if (start + 1) % ratio != 0:
                return None
            pooled = (kv_state * score_state.softmax(dim = 1)).sum(dim = 1, keepdim = True)
        return self.compressor_norm.forward(pooled.to(dtype = x.dtype), params)

    def _index_body(self, x, qr, latent, start, offset, runtime, end, ratio, batch, seqlen, params):
        # Later layers reuse the index source's top-k. The offset is already applied: every
        # layer has the same window, so the published indices address the concatenated KV.
        if not self.is_index_source:
            published = runtime.get("topk_idxs", x.device)
            if published is None:
                return x.new_empty(batch, seqlen, 0, dtype = torch.int32)
            return published.to(dtype = torch.int32)
        if self.is_kv_source and latent is not None:
            if start == 0:
                positions = torch.arange(latent.shape[1], device = x.device) * ratio
            else:
                positions = torch.tensor([start + 1 - ratio], device = x.device)
            k = self.idx_k_norm.forward(self.idx_wk.forward(latent, params), params)
            k = M.apply_rotary_tail(k, self._freqs(positions), self.rope_dim)
            k = M.fp4_fake_quant(k, 32, "e8m0")
            cache = runtime.get("index_k", x.device)
            dest = start // ratio
            cache = _grow(cache, batch, dest + k.shape[1], k.shape[-1], k, 0.0)
            cache[:, dest:dest + k.shape[1]] = k
            runtime.put("index_k", cache)
        index_k = runtime.get("index_k", x.device)
        n_pos = end // ratio if ratio else 0
        if index_k is None or n_pos == 0:
            idxs = x.new_empty(batch, seqlen, 0, dtype = torch.int32)
            runtime.put("topk_idxs", idxs)
            return idxs
        index_k = index_k[:, :n_pos]
        q = self.idx_wq_b.forward(qr, params).unflatten(-1, (self.index_n_heads, self.index_head_dim))
        qpos = torch.arange(start, end, device = x.device)
        q = M.apply_rotary_tail(q, self._freqs(qpos), self.rope_dim)
        q = M.fp4_fake_quant(q, 32, "e8m0")
        scale = self.index_head_dim ** -0.5
        weights = self.idx_weights.forward(x, params).float() * (scale * self.index_n_heads ** -0.5)
        scores = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
        scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim = 2)
        if start == 0:
            lens = (torch.arange(1, seqlen + 1, device = x.device) // ratio).unsqueeze(-1)
            scores = scores.masked_fill(torch.arange(n_pos, device = x.device) >= lens, float("-inf"))
            compress_lens = lens.squeeze(-1)
        else:
            compress_lens = end // ratio
            # a decode query sees every compressed position that exists; the mask is a no-op
        if self.is_candidate_source:
            candidates = M.select_candidate_blocks(
                scores, compress_lens, self.candidate_topk_blocks, self.candidate_block_size,
            )
            runtime.put("candidates", candidates)
        elif self.uses_candidates:
            candidates = runtime.get("candidates", x.device)
            if candidates is not None:
                scores = scores.masked_fill(~candidates[..., :scores.shape[-1]], float("-inf"))
        take = min(self.index_topk, n_pos)
        chosen = scores.topk(take, dim = -1, sorted = False).indices.sort(dim = -1).values
        if start == 0:
            visible = chosen < compress_lens.view(1, seqlen, 1)
        else:
            visible = chosen < compress_lens
        idxs = torch.where(visible, chosen + offset, torch.full_like(chosen, -1)).to(torch.int32)
        runtime.put("topk_idxs", idxs)
        return idxs

    def _compress(self, x, qr, start, offset, runtime, params) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, _ = x.shape
        ratio = self.compress_ratio
        compress_len = (start + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self._latent(x, params, runtime, start)
        idxs = self._index_body(
            x, qr, latent, start, offset, runtime,
            start + seqlen, ratio, batch, seqlen, params,
        ) if self.compress_ratio else x.new_empty(batch, seqlen, 0, dtype = torch.int32)
        if latent is not None:
            if start == 0:
                positions = torch.arange(latent.shape[1], device = x.device) * ratio
            else:
                positions = torch.tensor([start + 1 - ratio], device = x.device)
            latent = M.apply_rotary_tail(latent, self._freqs(positions), self.rope_dim)
            latent = M.fp4_fake_quant(latent, 16, "e4m3")
            cache = _grow(runtime.get("compress_kv", x.device), batch, start // ratio + latent.shape[1], latent.shape[-1], latent, 0.0)
            dest = start // ratio
            cache[:, dest:dest + latent.shape[1]] = latent
            runtime.put("compress_kv", cache)
        cache = runtime.get("compress_kv", x.device)
        if cache is None or compress_len == 0:
            empty = x.new_empty(batch, 0, self.head_dim)
            empty_i = x.new_empty(batch, seqlen, 0, dtype = torch.int32)
            return empty, empty_i
        return cache[:, :compress_len], idxs

    def _project_out(self, o: torch.Tensor, params: dict) -> torch.Tensor:
        batch, seqlen, _, _ = o.shape
        o = o.view(batch, seqlen, self.o_groups, -1)
        o = M.fp8_fake_quant(o, 32)
        parts = [self.wo_a[g].forward(o[:, :, g].half(), params) for g in range(self.o_groups)]
        return self.wo_b.forward(torch.cat(parts, dim = -1), params)

    def _forward_trunk(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        runtime = bind_runtime(params, x.shape[0])
        start = start_position(params)
        batch, seqlen, _ = x.shape
        positions = torch.arange(start, start + seqlen, device = x.device)
        freqs = self._freqs(positions)
        qr = self.q_norm.forward(self.q_a.forward(x, params), params)
        q = self.q_b.forward(qr, params).unflatten(-1, (self.n_heads, self.head_dim))
        q = M.apply_rotary_tail(q, freqs, self.rope_dim)
        kv = self.kv_norm.forward(self.wkv.forward(x, params), params)
        kv = M.apply_rotary_tail(kv, freqs, self.rope_dim)
        kv = M.fp8_fake_quant(kv, 32)
        window_kv, idxs = self._window(runtime, kv, start)
        if self.compress_ratio:
            compressed, cidx = self._compress(x, qr, start, window_kv.shape[1], runtime, params)
            kv = torch.cat([window_kv, compressed], dim = 1)
            idxs = torch.cat([idxs, cidx], dim = -1)
        else:
            kv = window_kv
        out = M.sparse_attn(q, kv, self.attn_sink, idxs, self.head_dim ** -0.5)
        out = M.apply_rotary_tail(out, freqs, self.rope_dim, inverse = True)
        return self._project_out(out, params)

    def _forward_dspark(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Draft attention. Prefill (start 0) only seeds the window from the trunk hidden."""
        runtime = bind_runtime(params, x.shape[0])
        start = start_position(params)
        main_x = params["dsv41_main_x"]
        batch, main_len, _ = main_x.shape
        main_pos = torch.arange(start, start + main_len, device = x.device)
        main_freqs = self._freqs(main_pos)
        main_kv = self.kv_norm.forward(self.wkv.forward(main_x, params), params)
        main_kv = M.apply_rotary_tail(main_kv, main_freqs, self.rope_dim)
        main_kv = M.fp8_fake_quant(main_kv, 32)
        self._window(runtime, main_kv, start)
        if start == 0:
            return x
        block = x.shape[1]
        positions = torch.arange(start + main_len, start + main_len + block, device = x.device)
        freqs = self._freqs(positions)
        qr = self.q_norm.forward(self.q_a.forward(x, params), params)
        q = self.q_b.forward(qr, params).unflatten(-1, (self.n_heads, self.head_dim))
        q = M.apply_rotary_tail(q, freqs, self.rope_dim)
        kv = self.kv_norm.forward(self.wkv.forward(x, params), params)
        kv = M.apply_rotary_tail(kv, freqs, self.rope_dim)
        kv = M.fp8_fake_quant(kv, 32)
        cache = runtime.get(f"win.{self.layer_idx}", x.device)
        topk = M.dspark_topk_idxs(self.window, batch, block, start, device = x.device)
        cat = torch.cat([cache, kv], dim = 1)
        out = M.sparse_attn(q, cat, self.attn_sink, topk, self.head_dim ** -0.5)
        out = M.apply_rotary_tail(out, freqs, self.rope_dim, inverse = True)
        return self._project_out(out, params)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        if self.dspark:
            return self._forward_dspark(x, params)
        return self._forward_trunk(x, params)


class DSV41Block(Module):
    """One trunk or DSpark block. mHC pre computed here is consumed by the next sublayer."""

    def __init__(
        self,
        config,
        key: str,
        layer_idx: int,
        attn: DSV41Attention,
        mlp: BlockSparseMLP,
        engram: DSV41Engram | None = None,
        main_proj: Linear | None = None,
        main_norm: RMSNorm | None = None,
        draft_norm: RMSNorm | None = None,
        markov_embed: Module | None = None,
        markov_head: Linear | None = None,
        confidence: Linear | None = None,
        target_layers: tuple[int, ...] = (),
        dspark: bool = False,
    ):
        super().__init__(config, key, None)
        self.layer_idx = layer_idx
        self.attn = attn
        self.mlp = mlp
        self.engram = engram
        self.main_proj = main_proj
        self.main_norm = main_norm
        self.draft_norm = draft_norm
        self.markov_embed = markov_embed
        self.markov_head = markov_head
        self.confidence = confidence
        self.target_layers = set(target_layers)
        self.dspark = dspark
        self.hc_mult = config.hc_mult
        self.attn_norm = RMSNorm(config, f"{key}.attn_norm", config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config, f"{key}.ffn_norm", config.rms_norm_eps)
        self.attn_hc = HyperConnection(
            config, f"{key}.hc_attn", config.hc_mult, config.hidden_size,
            config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps,
        )
        self.ffn_hc = HyperConnection(
            config, f"{key}.hc_ffn", config.hc_mult, config.hidden_size,
            config.hc_sinkhorn_iters, config.hc_eps, config.rms_norm_eps,
        )
        for module in (
            engram, main_proj, main_norm, self.attn_hc, self.attn_norm, attn,
            self.ffn_hc, self.ffn_norm, mlp, draft_norm, markov_embed, markov_head, confidence,
        ):
            self.register_submodule(module)
        self.num_slices = mlp.num_slices if mlp else 1

    @override
    def optimizer_targets(self):
        return [self.attn.optimizer_targets(), self.mlp.optimizer_targets()]

    def _mix(self, streams: torch.Tensor, hc: HyperConnection):
        flat = streams.flatten(2).float()
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim = True) + hc.rms_eps)
        mixes = F.linear(flat, hc.fn) * rsqrt
        return M.hc_split_sinkhorn(mixes, hc.scale, hc.base, hc.hc_mult, hc.sinkhorn_iters, hc.hc_eps)

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        runtime = bind_runtime(params, x.shape[0])
        start = start_position(params)
        if self.engram is not None:
            x = self.engram.forward(x, params)
        if self.layer_idx in self.target_layers and not self.dspark:
            runtime.put(f"tap.{self.layer_idx}", x.float().mean(dim = 2))
        if self.main_proj is not None:
            hidden = params["dsv41_main_hidden"]
            params["dsv41_main_x"] = self.main_norm.forward(self.main_proj.forward(hidden, params), params)
        if self.dspark and start == 0:
            self.attn.forward(x, params)
            return x
        pre = runtime.get("pre_mix", x.device)
        if pre is None:
            pre = M.identity_pre_mix(x, self.hc_mult)
        elif pre.shape[1] != x.shape[1]:
            pre = pre[:, :x.shape[1]]
        residual = x
        attn_pre, attn_post, attn_comb = self._mix(x, self.attn_hc)
        y = M.hc_pre(x, pre)
        y = self.attn_norm.forward(y, params, out_dtype = torch.half)
        y = self.attn.forward(y, params)
        x = M.hc_post(y, residual, attn_post, attn_comb)
        residual = x
        ffn_pre, ffn_post, ffn_comb = self._mix(x, self.ffn_hc)
        y = M.hc_pre(x, attn_pre)
        y = self.ffn_norm.forward(y, params, out_dtype = torch.half)
        y = self.mlp.forward(y, params)
        x = M.hc_post(y, residual, ffn_post, ffn_comb)
        runtime.put("pre_mix", ffn_pre)
        return x


class DSV41Collapse(Module):
    """Weightless final mix: the last FFN's pre, then the caller runs norm and head."""

    def __init__(self, config, hc_mult: int, target_layers: tuple[int, ...] = ()):
        super().__init__(config, "hc_collapse", None)
        self.hc_mult = hc_mult
        self.target_layers = tuple(target_layers)

    @override
    def optimizer_targets(self):
        return []

    @override
    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        runtime = bind_runtime(params, x.shape[0])
        pre = runtime.get("pre_mix", x.device)
        if pre is None:
            pre = M.identity_pre_mix(x, self.hc_mult)
        y = M.hc_pre(x, pre)
        taps = []
        for layer in self.target_layers:
            tap = runtime.get(f"tap.{layer}", x.device)
            if tap is not None:
                taps.append(tap)
        if taps:
            params["dsv41_main_hidden"] = torch.cat(taps, dim = -1)
        return y.to(dtype = out_dtype or torch.half)
