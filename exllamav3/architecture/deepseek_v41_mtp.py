from __future__ import annotations

import torch
from typing_extensions import override

from ..model.model import Model
from ..modules import Embedding, Linear, RMSNorm
from .dsv41_build import build_block

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .deepseek_v41 import DeepseekV41Config


class DeepseekV41MTPModel(Model):
    """DSpark blocks under ``mtp.*``.

    Weights are converted (uncalibrated, ``mtp_bits``). Speculative decoding is not wired
    into the generator: a forward with token ids raises. The trunk head is shared and is
    not duplicated here.
    """

    def __init__(self, config: DeepseekV41Config, **kwargs):
        super().__init__(config, **kwargs)
        if config.num_mtp_layers <= 0:
            raise RuntimeError("DeepseekV41 MTP component requires mtp blocks in compress_ratios")
        h = config.hidden_size
        n_taps = max(len(config.dspark_target_layer_ids), 1)
        last = config.num_mtp_layers - 1
        self.first_block_idx = 0
        for idx, ratio in enumerate(config.mtp_ratios[:config.num_mtp_layers]):
            if ratio != 0:
                raise ValueError(f"DeepseekV41 MTP block {idx} has compress_ratio {ratio}, expected 0")
            key = f"mtp.{idx}"
            main_proj = main_norm = draft_norm = None
            markov_embed = markov_head = confidence = None
            if idx == 0:
                main_proj = Linear(
                    config = config,
                    key = "mtp.0.main_proj",
                    in_features = h * n_taps,
                    out_features = h,
                    qmap = "block.attn",
                    out_dtype = torch.half,
                    qbits_key = "mtp_bits",
                    trim_padded_out = True,
                )
                main_norm = RMSNorm(config, "mtp.0.main_norm", config.rms_norm_eps)
            if idx == last:
                draft_norm = RMSNorm(config, f"mtp.{last}.norm", config.rms_norm_eps, out_dtype = torch.half)
                markov_embed = Embedding(
                    config = config,
                    key = f"mtp.{last}.markov_head.embed",
                    vocab_size = config.vocab_size,
                    hidden_size = config.dspark_markov_rank,
                )
                markov_head = Linear(
                    config = config,
                    key = f"mtp.{last}.markov_head.head",
                    in_features = config.dspark_markov_rank,
                    out_features = config.vocab_size,
                    qmap = None,
                    out_dtype = torch.float,
                    pad_to = 1,
                )
                confidence = Linear(
                    config = config,
                    key = f"mtp.{last}.confidence_head.proj",
                    in_features = h + config.dspark_markov_rank,
                    out_features = 1,
                    qmap = None,
                    out_dtype = torch.float,
                    pad_to = 1,
                )
            self.modules.append(build_block(
                config,
                key,
                idx,
                ratio = 0,
                n_experts = config.dspark_n_routed_experts,
                n_experts_per_tok = config.dspark_num_experts_per_tok,
                qbits_key = "mtp_bits",
                dspark = True,
                kv_source = False,
                index_source = False,
                with_engram = False,
                main_proj = main_proj,
                main_norm = main_norm,
                draft_norm = draft_norm,
                markov_embed = markov_embed,
                markov_head = markov_head,
                confidence = confidence,
            ))
        self.last_kv_module_idx = len(self.modules) - 1
        self.caps.update({"supports_tp": False})

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        if input_ids is not None and input_ids.ndim == 2:
            raise NotImplementedError(
                "DeepSeek-V4.1 DSpark speculative decoding is not wired into the generator. "
                "MTP weights are still converted."
            )
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        return prompt
