"""Pure-torch DeepSeek-V4.1 math.

Transcribed from the official MIT reference (inference/model.py, kernel.py, engram.py).
No CUDA extension import, so CPU tests can load this file without building exllamav3_ext.
Fake-quant matches the reference formulas (E8M0 power-of-two scales, E4M3 scales, E2M1
levels). It is not bit-identical to the TileLang kernels' rounding.
"""
from __future__ import annotations

import math

import numpy as np
import torch


FP8_MAX = 448.0
FP4_MAX = 6.0
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
DEAD = -1


def ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for finite x > 0, via the float32 exponent. Exact powers stay exact."""
    x = x.float().contiguous().clamp_min(torch.finfo(torch.float32).tiny)
    bits = x.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127
    man = bits & 0x7FFFFF
    return exp + (man != 0).to(dtype = exp.dtype)


def e8m0_scale(amax: torch.Tensor, fp_max: float) -> torch.Tensor:
    """Power-of-two scale: 2^ceil(log2(amax / fp_max)), the reference fast_round_scale."""
    ratio = (amax.float() / fp_max).clamp_min(2.0 ** -126)
    return torch.exp2(ceil_log2(ratio))


def fp8_fake_quant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Block FP8 E4M3 with E8M0 scales, dequantized back to ``x.dtype``. Last dim is the block axis."""
    n = x.shape[-1]
    if n % block_size:
        return x
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(dim = -1).clamp_min(1e-4)
    scale = e8m0_scale(amax, FP8_MAX).unsqueeze(-1)
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(x.shape).to(dtype = x.dtype)


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    levels = x.new_tensor(_E2M1)
    idx = (x.abs().unsqueeze(-1) - levels).abs().argmin(dim = -1)
    return torch.copysign(levels[idx], x)


def fp4_fake_quant(x: torch.Tensor, block_size: int, scale_kind: str) -> torch.Tensor:
    """Block FP4 E2M1. ``scale_kind`` is ``e8m0`` (indexer) or ``e4m3`` (compressed KV)."""
    n = x.shape[-1]
    if n % block_size:
        return x
    xf = x.float().reshape(-1, n // block_size, block_size)
    amax = xf.abs().amax(dim = -1)
    if scale_kind == "e4m3":
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -9)
        scale = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    elif scale_kind == "e8m0":
        amax = amax.clamp_min(FP4_MAX * 2.0 ** -126)
        scale = e8m0_scale(amax, FP4_MAX)
    else:
        raise ValueError(scale_kind)
    scale = scale.unsqueeze(-1)
    q = _nearest_e2m1((xf / scale).clamp(-FP4_MAX, FP4_MAX)) * scale
    return q.reshape(x.shape).to(dtype = x.dtype)


def yarn_inv_freq(
    dim: int,
    base: float,
    factor: float,
    original_seq_len: int,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Inverse frequencies, length dim/2. ``original_seq_len == 0`` disables YaRN."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype = torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype = torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cis_at(positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Complex exponentials at ``positions`` (any shape). Last dim is dim/2."""
    ang = positions.float().unsqueeze(-1) * inv_freq.to(positions.device)
    return torch.polar(torch.ones_like(ang), ang)


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """GPT-J adjacent-pair rotation. ``x`` is [b, s, d] or [b, s, h, d]; ``freqs`` is [s, d/2]."""
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    if xc.ndim == 3:
        freqs = freqs.view(1, xc.size(1), xc.size(-1))
    else:
        freqs = freqs.view(1, xc.size(1), 1, xc.size(-1))
    y = torch.view_as_real(xc * freqs).flatten(-2)
    return y.to(dtype = x.dtype)


def apply_rotary_tail(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False) -> torch.Tensor:
    if rope_dim <= 0:
        return x
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    return torch.cat([head, apply_rotary(tail, freqs, inverse)], dim = -1)


def identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    pre = x.new_zeros(x.shape[0], x.shape[1], hc_mult, dtype = torch.float32)
    pre[:, :, 0] = 1.0
    return pre


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pre, post, comb. ``mixes`` is [..., (2 + hc) * hc], matching kernel.py hc_split_sinkhorn."""
    hc = hc_mult
    pre_w, post_w, comb_w = mixes.split([hc, hc, hc * hc], dim = -1)
    pre_b, post_b, comb_b = hc_base.split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = hc_scale.unbind(0)
    pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
    comb = (comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_s + comb_b.view(hc, hc))
    comb = torch.softmax(comb, dim = -1) + eps
    comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim = -1, keepdim = True) + eps)
        comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
    return pre, post, comb


def hc_pre(streams: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """[b, s, hc, d] x [b, s, hc] -> [b, s, d]."""
    y = (pre_mix.unsqueeze(-1) * streams.float()).sum(dim = 2)
    return y.to(dtype = streams.dtype)


def hc_post(y: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    """post ⊗ y + comb^T residual. Does not write ``residual`` in place."""
    mixed = torch.matmul(comb.transpose(-1, -2), residual.float())
    out = post.unsqueeze(-1) * y.float().unsqueeze(-2) + mixed
    return out.to(dtype = y.dtype)


def pool_groups(kv: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Softmax over the group axis. kv/score: [b, n_groups, ratio, d] -> [b, n_groups, d]."""
    return (kv * score.softmax(dim = 2)).sum(dim = 2)


def window_topk_idxs(window: int, bsz: int, seqlen: int, start: int, device=None) -> torch.Tensor:
    """Sliding-window slots. -1 is empty. Shape [b, s, k]."""
    if start == 0:
        end = torch.arange(seqlen, device = device).unsqueeze(1)
        k = min(seqlen, window)
        idxs = (end - window + 1).clamp(min = 0) + torch.arange(k, device = device)
        idxs = torch.where(idxs > end, torch.full_like(idxs, -1), idxs)
    else:
        oldest = start % window + 1
        idxs = torch.cat([
            torch.arange(oldest, window, device = device),
            torch.arange(oldest, device = device),
        ])
        idxs = torch.where(idxs > start, torch.full_like(idxs, -1), idxs)
        idxs = idxs.view(1, -1)
    return idxs.to(torch.int32).unsqueeze(0).expand(bsz, -1, -1).contiguous()


def dspark_topk_idxs(window: int, bsz: int, block: int, start: int, device=None) -> torch.Tensor:
    """Draft attention indices: filled window slots, then the draft block after the ring."""
    if start <= 0:
        raise ValueError("dspark topk is a decode index list")
    body = torch.arange(min(window, start + 1), device = device)
    draft = window + torch.arange(block, device = device)
    idxs = torch.cat([body, draft]).to(torch.int32)
    return idxs.view(1, 1, -1).expand(bsz, block, -1).contiguous()


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Gathered attention. ``attn_sink`` is an extra logit in already-scaled space.

    Rows whose every index is -1 return zeros (the kernel's -1e30 floor), not NaN.
    q: [b, s, h, d], kv: [b, n, d], sink: [h], idxs: [b, s, k].
    """
    b, s, h, d = q.shape
    k = topk_idxs.shape[-1]
    if k == 0 or kv.shape[1] == 0:
        return q.new_zeros(q.shape)
    safe = topk_idxs.clamp(min = 0).to(torch.long)
    flat = safe.reshape(b, s * k, 1).expand(b, s * k, d)
    gathered = kv.gather(1, flat).view(b, s, k, d)
    valid = topk_idxs >= 0
    scores = torch.einsum("bshd,bskd->bshk", q.float(), gathered.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    m = scores.amax(dim = -1)
    m = torch.where(torch.isfinite(m), m, torch.full_like(m, -1e30))
    w = torch.exp(scores - m.unsqueeze(-1))
    w = torch.where(valid.unsqueeze(1), w, torch.zeros_like(w))
    sum_exp = w.sum(dim = -1) + torch.exp(attn_sink.float().view(1, 1, -1) - m)
    out = torch.einsum("bshk,bskd->bshd", w, gathered.float()) / sum_exp.unsqueeze(-1)
    return out.to(dtype = q.dtype)


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Bool mask over compressed positions. The newest partial block is pinned in."""
    width = logits.shape[-1]
    pad = (-width) % block_size
    scores = torch.nn.functional.pad(logits, (0, pad), value = float("-inf"))
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim = -1)
    num_blocks = scores.shape[-1]
    if isinstance(compress_lens, int):
        last = (compress_lens - 1) // block_size
        pin = torch.arange(num_blocks, device = logits.device) == last
    else:
        last = (compress_lens - 1) // block_size
        pin = torch.arange(num_blocks, device = logits.device) == last.unsqueeze(-1)
    scores = scores.masked_fill(pin, float("inf"))
    take = min(topk_blocks, num_blocks)
    top = scores.topk(take, dim = -1)
    keep = torch.zeros_like(scores, dtype = torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim = -1)[..., :width]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    if candidate % 2 == 0:
        candidate += 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 2
    return candidate


def engram_primes(
    layer_ids: tuple[int, ...] | list[int],
    max_ngram_size: int,
    n_heads: int,
    vocab_size: int,
) -> tuple:
    """[layer][n-gram size index][head] prime moduli. Primes are never reused."""
    seen: set[int] = set()
    primes = []
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes = []
            current = vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return tuple(primes)


def hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    """Odd int64 multipliers, one row per engram layer. Seed is ``10007 * layer_id``."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // int(vocab_size)) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * int(layer_id))
        values = generator.integers(low = 0, high = bound, size = (max_ngram_size,), dtype = np.int64)
        rows.append(torch.tensor(values * 2 + 1, dtype = torch.int64))
    return torch.stack(rows)


def engram_offsets(primes: tuple) -> torch.Tensor:
    """Exclusive prefix sums of the flattened (n-gram, head) prime widths, per layer."""
    rows = []
    for layer in primes:
        flat = [p for per_ngram in layer for p in per_ngram]
        offsets = np.cumsum([0, *flat[:-1]])
        rows.append(torch.tensor(offsets, dtype = torch.int64))
    return torch.stack(rows)


def ngram_hash_ids(
    compressed: torch.Tensor,
    cache: torch.Tensor,
    start: int,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    multipliers: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Hash ids [B, L, n_layers, n_hash_cols]. Updates ``cache`` at ``start:start+L``.

    ``compressed`` is the mapped token id, or DEAD (-1) where the token is outside an n-gram.
    ``primes`` is [n_layers, max_ngram-1, n_heads].
    """
    batch, seqlen = compressed.shape
    cache = cache[:batch]
    cache[:, start:start + seqlen] = compressed
    positions = torch.arange(start, start + seqlen, device = compressed.device).expand(batch, seqlen)
    tokens = []
    blocked = torch.zeros_like(positions, dtype = torch.bool)
    max_ngram = multipliers.shape[-1]
    for shift in range(max_ngram):
        source = cache.gather(1, (positions - shift).clamp(min = 0))
        blocked = blocked | (positions < shift) | (source == DEAD)
        tokens.append(torch.where(blocked, torch.full_like(source, pad_id), source))
    tokens = torch.stack(tokens, dim = -1)
    products = tokens.unsqueeze(2) * multipliers.to(tokens.device)
    rolling = products[..., 0]
    hashes = []
    for i in range(1, max_ngram):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1].to(rolling.device))
    return torch.cat(hashes, dim = -1) + offsets.to(rolling.device)


def engram_gate(
    streams: torch.Tensor,
    key: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    clamp_value: float = 1e-6,
) -> torch.Tensor:
    """Per-(token, hc copy) signed-sqrt sigmoid gate. streams/key: [b, s, hc, d], weight: [hc, d]."""
    h = streams.float()
    key = key.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight.float() * key).sum(-1) * rstd * (h.shape[-1] ** -0.5)
    return torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))


def dequant_fp8_e8m0(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 E4M3 rows times per-block E8M0 scales. ``scale`` is uint8 (byte - 127) or a float scale."""
    w = weight.float().unflatten(-1, (-1, block))
    if scale.dtype == torch.uint8:
        s = torch.exp2(scale.float() - 127.0)
    else:
        s = scale.float()
    return (w * s.unsqueeze(-1)).flatten(-2)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """NFKC/NFD/strip-accents/lower/whitespace collapse. Same map the reference hashes with."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    # HuggingFace tokenizers implement len(); the raw Rust Tokenizer exposes get_vocab_size().
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    try:
        n_tokens = len(tokenizer)
    except TypeError:
        n_tokens = int(backend.get_vocab_size())
    key_to_new: dict[str, int] = {}
    lookup = [0] * n_tokens
    for token_id in range(n_tokens):
        text = backend.decode([token_id], skip_special_tokens = False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def moe_sqrtsoftplus_weights(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    route_scale: float,
    norm_topk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias chooses experts. Weights come from the unbiased scores, then (optional) renorm."""
    scores = torch.nn.functional.softplus(logits).sqrt()
    indices = (scores + bias).topk(topk, dim = -1).indices
    weights = scores.gather(1, indices)
    if norm_topk and topk > 1:
        weights = weights / (weights.sum(dim = -1, keepdim = True) + 1e-20)
    return weights * route_scale, indices
