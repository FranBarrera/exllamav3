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


def test_revision_is_selected_by_config_readme_or_path(tmp_path):
    layout = load_layout()
    assert layout.REVISION_1017.startswith("1bae0e10")
    assert layout.REVISION_0909.startswith("dba1be0a")
    bare = tmp_path / "checkpoint"
    bare.mkdir()
    assert layout.resolve_revision(str(bare), {}) == "1017"
    assert layout.resolve_revision(str(bare), {"dsv41_revision": "0909"}) == "0909"
    assert layout.resolve_revision(str(bare), {"text_config": {"dsv41_revision": "1017"}}) == "1017"
    readme = bare / "README.md"
    readme.write_text("# DeepSeek-V4.1-Flash\n\nSeptember weights.\n")
    assert layout.resolve_revision(str(bare), {}) == "0909"
    readme.write_text("# DeepSeek-V4.1-Flash-1017\n\nrevision 1bae0e10\n")
    assert layout.resolve_revision(str(bare), {}) == "1017"
    # explicit key beats the README
    assert layout.resolve_revision(str(bare), {"dsv41_revision": "0909"}) == "0909"
    named = tmp_path / "DeepSeek-V4.1-Flash-0909"
    named.mkdir()
    assert layout.resolve_revision(str(named), {}) == "0909"


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
