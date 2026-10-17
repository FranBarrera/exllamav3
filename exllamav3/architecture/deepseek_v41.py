from __future__ import annotations

from types import SimpleNamespace

import torch
from typing_extensions import override

from ..cache.recurrent_util import prepare_for_recurrence
from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Embedding, ExpandStreams, Linear, RMSNorm
from ..modules.attn import prepare_for_attn
from ..modules.dsv41 import DSV41Collapse, DSV41State
from .deepseek_v41_mtp import DeepseekV41MTPModel
from .deepseek_v41_vision import DeepseekV41VisionModel
from .dsv41_build import build_block
from .dsv41_layout import passthrough_engram_keys, resolve_revision


def _text(config: Config, expected, name: str, default = no_default):
    """Read a text hyperparameter from the nested HF config or a flat tiny config."""
    return config.read_cfg(expected, [f"text_config->{name}", name], default)


def _first(config: Config, expected, names: list[str], default = no_default):
    keys = []
    for name in names:
        keys.append(f"text_config->{name}")
        keys.append(name)
    return config.read_cfg(expected, keys, default)


def read_deepseek_v41_vision(config: Config) -> SimpleNamespace | None:
    raw = config.read_cfg(dict, "vision_config", None)
    if not raw:
        return None

    def pick(*names, default = None):
        for name in names:
            if name in raw and raw[name] is not None:
                return raw[name]
        return default

    n_layers = int(pick("num_hidden_layers", "vision_n_layers", default = 0) or 0)
    if n_layers <= 0:
        return None
    dim = int(pick("hidden_size", "vision_dim", default = 1024))
    n_heads = int(pick("num_attention_heads", "vision_n_heads", default = 16))
    return SimpleNamespace(
        n_layers = n_layers,
        dim = dim,
        n_heads = n_heads,
        head_dim = dim // n_heads,
        inter_dim = int(pick("intermediate_size", "vision_inter_dim", default = 2816)),
        patch_size = int(pick("patch_size", "vision_patch_size", default = 14)),
        rope_theta = float(pick("rope_theta", "vision_rope_theta", default = 10000.0)),
        downsample_ratio = int(pick("downsample_ratio", "vision_downsample_ratio", default = 3)),
        num_channels = 3,
        rms_norm_eps = 1e-6,
    )


class DeepseekV41Config(Config):
    arch_string = "DeepseekV41ForCausalLM"

    def __init__(self, directory: str, **kwargs):
        super().__init__(
            directory,
            {"text": DeepseekV41Model, "mtp": DeepseekV41MTPModel},
            **kwargs,
        )

        self.hidden_size = _text(self, int, "hidden_size", no_default)
        self.num_q_heads = _text(self, int, "num_attention_heads", no_default)
        self.num_kv_heads = _text(self, int, "num_key_value_heads", 1)
        if self.num_kv_heads != 1:
            raise ValueError("DeepseekV41: expected shared-KV MQA (num_key_value_heads == 1)")
        self.head_dim = _text(self, int, "head_dim", 512)
        self.qk_rope_head_dim = _text(self, int, "qk_rope_head_dim", 64)
        self.q_lora_rank = _text(self, int, "q_lora_rank", no_default)
        self.o_groups = _text(self, int, "o_groups", 8)
        self.o_lora_rank = _text(self, int, "o_lora_rank", 1024)
        self.sliding_window = _text(self, int, "sliding_window", 128)
        self.index_n_heads = _text(self, int, "index_n_heads", 32)
        self.index_head_dim = _text(self, int, "index_head_dim", 128)
        self.index_topk = _text(self, int, "index_topk", 512)

        self.num_hidden_layers = _text(self, int, "num_hidden_layers", no_default)
        ratios = _text(self, list, "compress_ratios", no_default)
        self.compress_ratios = [int(r) for r in ratios]
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError(
                f"compress_ratios has {len(self.compress_ratios)} entries, "
                f"need at least {self.num_hidden_layers}"
            )
        self.layer_ratios = self.compress_ratios[:self.num_hidden_layers]
        self.mtp_ratios = self.compress_ratios[self.num_hidden_layers:]

        self.kv_source_layers = [int(i) for i in _first(
            self, list, ["kv_source_layer_ids", "kv_source_layers"], [])]
        self.index_source_layers = [int(i) for i in _first(
            self, list, ["index_source_layer_ids", "index_source_layers"], [])]
        self.candidate_source_layer = int(_first(
            self, int, ["candidate_source_layer_id", "candidate_source_layer"], -1))
        self.candidate_topk_blocks = _text(self, int, "candidate_topk_blocks", 2048)
        self.candidate_block_size = _text(self, int, "candidate_block_size", 8)

        self.hc_mult = _text(self, int, "hc_mult", 4)
        self.hc_sinkhorn_iters = _text(self, int, "hc_sinkhorn_iters", 20)
        self.hc_eps = _text(self, float, "hc_eps", 1e-6)

        self.assert_cfg(str, ["text_config->scoring_func", "scoring_func"], "sqrtsoftplus", optional = True)
        self.assert_cfg(str, ["text_config->topk_method", "topk_method"], "noaux_tc", optional = True)
        self.moe_intermediate_size = _text(self, int, "moe_intermediate_size", no_default)
        self.num_experts = _text(self, int, "n_routed_experts", no_default)
        self.num_experts_per_tok = _text(self, int, "num_experts_per_tok", no_default)
        self.num_shared_experts = _text(self, int, "n_shared_experts", 1)
        self.routed_scaling_factor = _text(self, float, "routed_scaling_factor", 1.0)
        self.swiglu_limit = _text(self, float, "swiglu_limit", 10.0)

        self.rms_norm_eps = _text(self, float, "rms_norm_eps", 1e-6)
        self.rope_theta = _text(self, float, "rope_theta", 10000.0)
        self.compress_rope_theta = _text(self, float, "compress_rope_theta", 160000.0)
        scaling = _text(self, dict, "rope_scaling", None) or {}
        self.rope_scaling = scaling or None
        self.yarn = {
            "factor": float(scaling.get("factor", 1.0)),
            "beta_fast": float(scaling.get("beta_fast", 32)),
            "beta_slow": float(scaling.get("beta_slow", 1)),
            "original_seq_len": int(scaling.get("original_max_position_embeddings", 0) or 0),
        }

        self.tie_word_embeddings = _text(self, bool, "tie_word_embeddings", False)

        self.engram_layer_ids = [int(i) for i in _text(self, list, "engram_layer_ids", [])]
        self.engram_num_embeddings = [int(n) for n in _text(self, list, "engram_num_embeddings", [])]
        self.engram_max_ngram_size = _text(self, int, "engram_max_ngram_size", 4)
        self.engram_vocab_size = _text(self, int, "engram_vocab_size", 16000000)
        self.engram_n_heads = _text(self, int, "engram_n_heads", 8)
        self.engram_head_dim = _text(self, int, "engram_head_dim", 256)
        self.engram_pad_id = int(_first(
            self, int, ["engram_pad_token_id", "engram_pad_id"], 2))
        self.engram_compressed_vocab_size = _text(self, int, "engram_compressed_vocab_size", 99092)

        self.dspark_block_size = _text(self, int, "dspark_block_size", 0)
        self.dspark_noise_token_id = _text(self, int, "dspark_noise_token_id", 0)
        self.dspark_markov_rank = _text(self, int, "dspark_markov_rank", 256)
        self.dspark_target_layer_ids = [int(i) for i in _text(self, list, "dspark_target_layer_ids", [])]
        self.dspark_n_routed_experts = _text(self, int, "dspark_n_routed_experts", self.num_experts)
        self.dspark_num_experts_per_tok = int(_first(
            self, int, ["dspark_num_experts_per_tok", "dspark_n_activated_experts"], 3))
        self.block_size = self.dspark_block_size + 1

        self.num_mtp_layers = len(self.mtp_ratios)
        if self.num_mtp_layers == 0 or not any(
            self.stc.has_tensor(f"mtp.0.attn.wkv.{t}") for t in ("weight", "trellis")
        ):
            self.num_mtp_layers = 0
            del self.model_classes["mtp"]

        self.vision = read_deepseek_v41_vision(self)
        if self.vision is not None and any(
            self.stc.has_tensor(f"vision.patch_embed.proj.{t}") for t in ("weight", "trellis")
        ):
            self.model_classes["vision"] = DeepseekV41VisionModel
        else:
            self.vision = None

        # 1017 vs 0909. Same config.json; the forward differs. See dsv41_layout.resolve_revision.
        self.dsv41_revision = resolve_revision(directory, self.config_dict)
        self.wo_a_act_quant = self.dsv41_revision == "1017"
        self.publish_index_k_always = self.dsv41_revision == "1017"


class DeepseekV41Model(Model):
    config_class = DeepseekV41Config

    @staticmethod
    @override
    def get_additional_compiled_tensors(config: DeepseekV41Config) -> dict:
        """Engram hash tables. They are not Linears, so compile copies them as extras.

        Tables at or above the stream-copy threshold are written through without being
        loaded. ``--ngram_bits`` does not apply: that path only quantizes NGramEmbedding.
        """
        found = {}
        for key in sorted(passthrough_engram_keys(config.config_dict)):
            if config.stc.has_tensor(key):
                found.update(config.stc.list_tensors(prefix = key))
        return found

    def __init__(self, config: DeepseekV41Config, **kwargs):
        super().__init__(config, **kwargs)
        self.modules += [
            Embedding(
                config = config,
                key = "embed",
                vocab_size = config.vocab_size,
                hidden_size = config.hidden_size,
            ),
            ExpandStreams(
                config = config,
                key = "hc_expand",
                hc_mult = config.hc_mult,
            ),
        ]
        self.first_block_idx = len(self.modules)
        engram_layers = set(config.engram_layer_ids)
        kv_sources = set(config.kv_source_layers)
        index_sources = set(config.index_source_layers)
        for idx, ratio in enumerate(config.layer_ratios):
            self.modules.append(build_block(
                config,
                f"layers.{idx}",
                idx,
                ratio = ratio,
                n_experts = config.num_experts,
                n_experts_per_tok = config.num_experts_per_tok,
                qbits_key = "bits",
                dspark = False,
                kv_source = idx in kv_sources,
                index_source = idx in index_sources,
                with_engram = idx in engram_layers,
            ))
        self.last_kv_module_idx = len(self.modules) - 1

        head_alt_key = None
        if config.tie_word_embeddings and not config.stc.has_tensor("head.weight"):
            head_alt_key = "embed"

        self.modules += [
            DSV41Collapse(
                config = config,
                hc_mult = config.hc_mult,
                target_layers = tuple(config.dspark_target_layer_ids),
            ),
            RMSNorm(
                config = config,
                key = "norm",
                rms_norm_eps = config.rms_norm_eps,
                out_dtype = torch.half,
            ),
            Linear(
                config = config,
                key = "head",
                qbits_key = "head_bits",
                alt_key = head_alt_key,
                in_features = config.hidden_size,
                out_features = config.vocab_size,
                qmap = "block",
                caps = {"logits_output": True},
            ),
        ]
        self.logit_layer_idx = len(self.modules) - 1
        self.calibration_all_experts = True
        # Shared KV, the delayed mHC pre-mix and Engram's token cache live in one bag.
        # There is no paged KV allocation and no in-place speculative rewind.
        self.caps.update({
            "recurrent_states": True,
            "supports_tp": False,
            "checkpoint_quant_preserve": True,
            "default_recurrent_checkpoint_interval": 2048,
        })
        self.recurrent_state_cls = DSV41State

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        params["input_ids"] = input_ids
        input_ids = prepare_for_attn(input_ids, params)
        params["input_ids"] = input_ids
        prepare_for_recurrence(input_ids, params, self)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        p = ""
        if system_prompt:
            p += f"{system_prompt}\n\n"
        p += f"<|User|>{prompt}<|Assistant|>"
        return p
