"""Shared trunk / DSpark block constructor for DeepSeek-V4.1.

Kept out of the model classes so the text model and the MTP component can both build a
block without importing each other.
"""
from __future__ import annotations

import torch

from ..modules import BlockSparseMLP, Embedding, GatedMLP, Linear, RMSNorm
from ..modules.dsv41 import DSV41Attention, DSV41Block, DSV41Engram


def build_block(
    config,
    key: str,
    layer_idx: int,
    *,
    ratio: int,
    n_experts: int,
    n_experts_per_tok: int,
    qbits_key: str,
    dspark: bool,
    kv_source: bool,
    index_source: bool,
    with_engram: bool,
    main_proj: Linear | None = None,
    main_norm: RMSNorm | None = None,
    draft_norm: RMSNorm | None = None,
    markov_embed: Embedding | None = None,
    markov_head: Linear | None = None,
    confidence: Linear | None = None,
):
    attn = DSV41Attention(
        config = config,
        key = f"{key}.attn",
        layer_idx = layer_idx,
        hidden_size = config.hidden_size,
        num_q_heads = config.num_q_heads,
        head_dim = config.head_dim,
        rope_dim = config.qk_rope_head_dim,
        q_lora_rank = config.q_lora_rank,
        o_groups = config.o_groups,
        o_lora_rank = config.o_lora_rank,
        window = config.sliding_window,
        compress_ratio = ratio,
        is_kv_source = kv_source,
        is_index_source = index_source,
        index_n_heads = config.index_n_heads,
        index_head_dim = config.index_head_dim,
        index_topk = config.index_topk,
        candidate_source = config.candidate_source_layer,
        candidate_topk_blocks = config.candidate_topk_blocks,
        candidate_block_size = config.candidate_block_size,
        rope_theta = config.rope_theta,
        compress_rope_theta = config.compress_rope_theta,
        yarn = config.yarn,
        rms_norm_eps = config.rms_norm_eps,
        qmap = "block.attn",
        qbits_key = qbits_key,
        select_hq_bits = 2,
        dspark = dspark,
    )
    engram = None
    if with_engram:
        engram = DSV41Engram(
            config = config,
            key = f"{key}.engram",
            layer_idx = layer_idx,
            hidden_size = config.hidden_size,
            hc_mult = config.hc_mult,
            qbits_key = qbits_key,
        )
    mlp = BlockSparseMLP(
        config = config,
        key = f"{key}.ffn",
        hidden_size = config.hidden_size,
        intermediate_size = config.moe_intermediate_size,
        num_experts = n_experts,
        num_experts_per_tok = n_experts_per_tok,
        key_up = "experts.{expert_idx}.w3",
        key_gate = "experts.{expert_idx}.w1",
        key_down = "experts.{expert_idx}.w2",
        key_routing_gate = "gate",
        key_e_score_bias = "gate.bias",
        key_e_score_bias_vl = "gate.bias_vl",
        qmap = "block.mlp",
        interm_dtype = torch.half,
        out_dtype = torch.float,
        activation_fn = "silu",
        act_limit = config.swiglu_limit,
        router_type = "sqrtsp",
        routed_scaling_factor = config.routed_scaling_factor,
        qbits_key = qbits_key,
        shared_experts = GatedMLP(
            config = config,
            key = f"{key}.ffn.shared_experts",
            hidden_size = config.hidden_size,
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts,
            key_up = "w3",
            key_gate = "w1",
            key_down = "w2",
            qmap = "block.mlp",
            out_dtype = torch.float,
            activation_fn = "silu",
            act_limit = config.swiglu_limit,
            select_hq_bits = 2,
            qbits_key = qbits_key,
        ),
    )
    return DSV41Block(
        config = config,
        key = key,
        layer_idx = layer_idx,
        attn = attn,
        mlp = mlp,
        engram = engram,
        main_proj = main_proj,
        main_norm = main_norm,
        draft_norm = draft_norm,
        markov_embed = markov_embed,
        markov_head = markov_head,
        confidence = confidence,
        target_layers = tuple(config.dspark_target_layer_ids),
        dspark = dspark,
    )
