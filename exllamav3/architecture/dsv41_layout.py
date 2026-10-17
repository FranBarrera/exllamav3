"""Checkpoint layout for DeepSeek-V4.1.

Pure Python (no torch, no extension). The architecture builds modules from the same
rules, and tests compare the resulting tensor names with the Flash-1017 safetensors
index. Names here are the source checkpoint names. Grouped ``wo_a`` is one source
tensor; conversion splits it into ``wo_a.slice.{g}`` EXL3 tensors.
"""
from __future__ import annotations

import os
import re

# Hugging Face commits. config.json does not record which one a directory is.
REVISION_1017 = "1bae0e1088993b9b82de08f59de5be5a34e42170"
REVISION_0909 = "dba1be0a40aa45a94ad051997016db3960a90277"

_LAYER = re.compile(r"layers\.\d+")
_MTP = re.compile(r"mtp\.\d+")
_EXPERT = re.compile(r"experts\.\d+")
_BLOCK = re.compile(r"blocks\.\d+")


def text_dict(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def vision_dict(cfg: dict) -> dict | None:
    return cfg.get("vision_config")


def collapse_key(key: str) -> str:
    key = _LAYER.sub("layers.N", key)
    key = _MTP.sub("mtp.N", key)
    key = _EXPERT.sub("experts.E", key)
    key = _BLOCK.sub("blocks.N", key)
    return key


def plan(cfg: dict) -> dict:
    """Resolved layer schedule. Accepts the nested HF config or a flat text config."""
    text = text_dict(cfg)
    n_layers = int(text["num_hidden_layers"])
    ratios = [int(r) for r in text["compress_ratios"]]
    if len(ratios) < n_layers:
        raise ValueError(f"compress_ratios has {len(ratios)} entries, need at least {n_layers}")
    kv = [int(i) for i in text.get("kv_source_layer_ids", text.get("kv_source_layers", []))]
    index = [int(i) for i in text.get("index_source_layer_ids", text.get("index_source_layers", []))]
    engram = [int(i) for i in text.get("engram_layer_ids", [])]
    n_mtp = len(ratios) - n_layers
    return {
        "n_layers": n_layers,
        "ratios": ratios[:n_layers],
        "mtp_ratios": ratios[n_layers:],
        "n_mtp": n_mtp,
        "kv_sources": kv,
        "index_sources": index,
        "engram_layers": engram,
        "n_experts": int(text["n_routed_experts"]),
        "n_mtp_experts": int(text.get("dspark_n_routed_experts", text["n_routed_experts"])),
        "o_groups": int(text.get("o_groups", 8)),
        "vision_layers": int(vision_dict(cfg)["num_hidden_layers"]) if vision_dict(cfg) else 0,
    }


def _with_scale(keys: set[str], stem: str, scale: bool) -> None:
    keys.add(stem + ".weight")
    if scale:
        keys.add(stem + ".scale")


def _attn_keys(keys: set[str], prefix: str, *, ratio: int, kv_source: bool, index_source: bool, scale_attn: bool) -> None:
    keys.add(f"{prefix}.attn.attn_sink")
    keys.add(f"{prefix}.attn.q_norm.weight")
    keys.add(f"{prefix}.attn.kv_norm.weight")
    for stem in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        _with_scale(keys, f"{prefix}.attn.{stem}", scale_attn)
    if kv_source:
        keys.add(f"{prefix}.attn.compressor.norm.weight")
        keys.add(f"{prefix}.attn.compressor.wkv.weight")
        if ratio > 1:
            keys.add(f"{prefix}.attn.compressor.wgate.weight")
    if index_source:
        keys.add(f"{prefix}.attn.indexer.weights_proj.weight")
        _with_scale(keys, f"{prefix}.attn.indexer.wq_b", True)
        if kv_source:
            keys.add(f"{prefix}.attn.indexer.wk.weight")
            keys.add(f"{prefix}.attn.indexer.k_norm.weight")


def _ffn_keys(keys: set[str], prefix: str, n_experts: int) -> None:
    keys.add(f"{prefix}.ffn_norm.weight")
    keys.add(f"{prefix}.ffn.gate.weight")
    keys.add(f"{prefix}.ffn.gate.bias")
    keys.add(f"{prefix}.ffn.gate.bias_vl")
    for proj in ("w1", "w2", "w3"):
        _with_scale(keys, f"{prefix}.ffn.shared_experts.{proj}", True)
        for e in range(n_experts):
            _with_scale(keys, f"{prefix}.ffn.experts.{e}.{proj}", True)
    for tag in ("attn", "ffn"):
        keys.add(f"{prefix}.hc_{tag}_fn")
        keys.add(f"{prefix}.hc_{tag}_base")
        keys.add(f"{prefix}.hc_{tag}_scale")
    keys.add(f"{prefix}.attn_norm.weight")


def _engram_keys(keys: set[str], layer: int) -> None:
    p = f"layers.{layer}.engram"
    _with_scale(keys, f"{p}.embed", True)
    _with_scale(keys, f"{p}.wkv", True)
    keys.add(f"{p}.q_weight")
    keys.add(f"{p}.k_weight")


def _vision_keys(keys: set[str], n_layers: int) -> None:
    _with_scale(keys, "vision.patch_embed.proj", False)
    keys.add("vision.patch_embed.proj.bias")
    for i in range(n_layers):
        b = f"vision.blocks.{i}"
        keys.add(f"{b}.norm1.weight")
        keys.add(f"{b}.norm2.weight")
        keys.add(f"{b}.attn.wqkv.weight")
        keys.add(f"{b}.attn.wqkv.bias")
        keys.add(f"{b}.attn.wo.weight")
        keys.add(f"{b}.attn.wo.bias")
        keys.add(f"{b}.mlp.w1.weight")
        keys.add(f"{b}.mlp.w2.weight")
    keys.add("vision.norm.weight")
    for stem in ("aligner.w1", "aligner.w2"):
        keys.add(stem + ".weight")
        keys.add(stem + ".bias")
    keys.add("image_start")
    keys.add("image_end")
    keys.add("image_newline")


def source_keys(cfg: dict, *, vision: bool = True, mtp: bool = True) -> set[str]:
    """Every tensor name in a V4.1 checkpoint described by ``cfg``."""
    p = plan(cfg)
    keys: set[str] = set()
    keys.add("embed.weight")
    keys.add("norm.weight")
    keys.add("head.weight")
    for i, ratio in enumerate(p["ratios"]):
        prefix = f"layers.{i}"
        _attn_keys(
            keys, prefix,
            ratio = ratio,
            kv_source = i in p["kv_sources"],
            index_source = i in p["index_sources"],
            scale_attn = True,
        )
        _ffn_keys(keys, prefix, p["n_experts"])
        if i in p["engram_layers"]:
            _engram_keys(keys, i)
    if mtp and p["n_mtp"]:
        last = p["n_mtp"] - 1
        for i, ratio in enumerate(p["mtp_ratios"]):
            prefix = f"mtp.{i}"
            _attn_keys(
                keys, prefix,
                ratio = ratio,
                kv_source = False,
                index_source = False,
                scale_attn = True,
            )
            _ffn_keys(keys, prefix, p["n_mtp_experts"])
        _with_scale(keys, "mtp.0.main_proj", True)
        keys.add("mtp.0.main_norm.weight")
        keys.add(f"mtp.{last}.norm.weight")
        keys.add(f"mtp.{last}.markov_head.embed.weight")
        keys.add(f"mtp.{last}.markov_head.head.weight")
        keys.add(f"mtp.{last}.confidence_head.proj.weight")
    if vision and p["vision_layers"]:
        _vision_keys(keys, p["vision_layers"])
    return keys


def resolve_revision(directory: str, config_dict: dict | None = None) -> str:
    """``"1017"`` or ``"0909"``.

    An explicit ``dsv41_revision`` key wins (top level or ``text_config``). Otherwise the
    checkpoint README, then the directory path. Official ``config.json`` files match, so a
    bare directory with neither hint defaults to 1017, the forward this tree targets.
    """
    cfg = config_dict or {}
    raw = cfg.get("dsv41_revision")
    text = cfg.get("text_config")
    if raw is None and isinstance(text, dict):
        raw = text.get("dsv41_revision")
    if raw is not None:
        token = str(raw).strip().lower()
        if token in {"1017", "1bae0e10", REVISION_1017}:
            return "1017"
        if token in {"0909", "dba1be0a", REVISION_0909}:
            return "0909"
        raise ValueError(
            f"dsv41_revision must be 1017 or 0909, got {raw!r}"
        )
    readme = os.path.join(directory, "README.md")
    if os.path.isfile(readme):
        with open(readme, encoding = "utf-8", errors = "ignore") as handle:
            head = handle.read(8000)
        if "DeepSeek-V4.1-Flash-1017" in head or "1bae0e10" in head:
            return "1017"
        if "DeepSeek-V4.1-Flash" in head:
            return "0909"
    path = directory.replace("\\", "/").lower()
    if "1017" in path or "1bae0e10" in path:
        return "1017"
    if "0909" in path or "dba1be0a" in path:
        return "0909"
    return "1017"


def passthrough_engram_keys(cfg: dict) -> set[str]:
    """Engram hash tables copied byte-for-byte (not quantized, not loaded whole)."""
    keys = set()
    for i in plan(cfg)["engram_layers"]:
        keys.add(f"layers.{i}.engram.embed.weight")
        keys.add(f"layers.{i}.engram.embed.scale")
    return keys
