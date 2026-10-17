"""Tensor names implied by the Flash-1017 config against the real safetensors index."""
import importlib.util
import json
import os
from collections import Counter

import pytest

pytestmark = pytest.mark.nogpu

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_layout():
    path = os.path.join(_ROOT, "exllamav3", "architecture", "dsv41_layout.py")
    spec = importlib.util.spec_from_file_location("dsv41_layout_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_keys_match_index_patterns():
    layout = load_layout()
    cfg = json.load(open(os.path.join(_FIX, "deepseek_v41_flash_config.json")))
    fixture = json.load(open(os.path.join(_FIX, "index_patterns.json")))
    keys = layout.source_keys(cfg)
    counts = Counter(layout.collapse_key(key) for key in keys)
    assert sum(counts.values()) == fixture["n_tensors"] == 96085
    assert dict(counts) == fixture["patterns"]
    passthrough = layout.passthrough_engram_keys(cfg)
    assert passthrough == {
        "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight",
        "layers.14.engram.embed.scale",
    }
    # compressor gate exists only where the ratio is greater than 1
    assert counts["layers.N.attn.compressor.wgate.weight"] == 3
    assert counts["layers.N.attn.compressor.wkv.weight"] == 4
    assert "hc_head_fn" not in counts
    assert "layers.N.ffn.gate.tid2eid" not in counts
