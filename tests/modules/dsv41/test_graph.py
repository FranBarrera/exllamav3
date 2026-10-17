"""Module tree vs the Flash-1017 index, under EXLLAMA_EXT_STUB (no CUDA build)."""
import json
import os

import pytest

from testlib.isolated import run_isolated

pytestmark = pytest.mark.nogpu

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_STUB = {"EXLLAMA_EXT_STUB": "1"}


def _walk(model):
    from exllamav3.modules import Embedding, Linear, RMSNorm
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.hyperconnections import HyperConnection
    keys = set()
    for module in model:
        if isinstance(module, Linear):
            keys.add((module.fkey or module.key) + ".weight")
        elif isinstance(module, RMSNorm) and not module.unweighted:
            keys.add(module.tensor_key)
        elif isinstance(module, Embedding):
            keys.add(module.key + ".weight")
        elif isinstance(module, HyperConnection):
            keys.add(module.key + "_fn")
            keys.add(module.key + "_base")
            keys.add(module.key + "_scale")
        elif isinstance(module, BlockSparseMLP):
            if module.e_score_correction_bias_key:
                keys.add(f"{module.key}.{module.e_score_correction_bias_key}")
            if module.e_score_bias_vl_key:
                keys.add(f"{module.key}.{module.e_score_bias_vl_key}")
        for extra in getattr(module, "extra_source_keys", ()) or ():
            keys.add(extra)
    return keys


def flash_keys():
    import os
    import shutil
    import torch
    from safetensors.torch import save_file
    from exllamav3.architecture.architectures import ARCHITECTURES
    from exllamav3.architecture.deepseek_v41_vision import ORPHAN_TENSORS
    from exllamav3.architecture.dsv41_layout import source_keys
    from exllamav3.model.model import Model
    from exllamav3.modules import Linear
    from exllamav3.modules.dsv41 import DSV41Attention, DSV41Engram

    assert "DeepseekV4ForCausalLM" in ARCHITECTURES
    assert "DeepseekV41ForCausalLM" in ARCHITECTURES
    assert len(ARCHITECTURES) > 40

    dst = "/tmp/dsv41_flash_graph"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    cfg = json.load(open(os.path.join(_FIX, "deepseek_v41_flash_config.json")))
    with open(os.path.join(dst, "config.json"), "w") as handle:
        json.dump(cfg, handle)
    save_file(
        {
            "mtp.0.attn.wkv.weight": torch.zeros(1, 1),
            "vision.patch_embed.proj.weight": torch.zeros(1, 1),
        },
        os.path.join(dst, "model.safetensors"),
    )
    config = ARCHITECTURES["DeepseekV41ForCausalLM"]["config_class"](dst)
    assert "mtp" in config.model_classes and "vision" in config.model_classes
    text = Model.from_config(config)
    mtp = Model.from_config(config, component = "mtp")
    vision = Model.from_config(config, component = "vision")
    walked = _walk(text) | _walk(mtp) | _walk(vision) | set(ORPHAN_TENSORS)
    expected = source_keys(cfg)
    extra = sorted(walked - expected)
    missing = sorted(expected - walked)
    assert not extra, extra[:20]
    assert missing and all(key.endswith(".scale") for key in missing), missing[:20]
    assert not any("engram.embed" in key for key in missing)
    slices = [m for m in text if isinstance(m, Linear) and ".wo_a.slice." in m.key]
    assert len(slices) == 40 * 8
    assert all(m.fkey.endswith(".wo_a") for m in slices)
    attn = next(m for m in text if isinstance(m, DSV41Attention) and m.is_index_source)
    assert attn.q_priority == 4
    assert attn.idx_weights.qmap is None
    # fixture directory has no README and no dsv41_revision, so the 1017 forward is the default
    assert config.dsv41_revision == "1017"
    assert attn.wo_a_act_quant is True
    assert attn.publish_index_k_always is True
    dspark_attn = next(m for m in mtp if isinstance(m, DSV41Attention))
    assert dspark_attn.wo_a_act_quant is True
    engram = next(m for m in text if isinstance(m, DSV41Engram))
    assert engram.weights_numel() == engram.wkv.weights_numel()
    assert engram.weights_numel() == 24 * 256 * 5120 * 5
    return {"keys": len(walked), "scales_only": len(missing)}


def tiny_strategy():
    import json
    import os
    import shutil
    from exllamav3.architecture.deepseek_v41 import DeepseekV41Config
    from exllamav3.conversion.allocation import create_q_strategy
    from exllamav3.model.model import Model

    cfg = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 64,
        "moe_intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 16,
        "o_lora_rank": 16,
        "o_groups": 2,
        "rms_norm_eps": 1e-6,
        "compress_ratios": [0, 2],
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "rope_scaling": {
            "factor": 1.0, "beta_fast": 32, "beta_slow": 1,
            "original_max_position_embeddings": 32,
        },
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "sliding_window": 4,
        "kv_source_layer_ids": [1],
        "index_source_layer_ids": [1],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 2,
        "candidate_source_layer_id": 1,
        "candidate_topk_blocks": 1,
        "candidate_block_size": 2,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 2,
        "hc_eps": 1e-6,
        "engram_layer_ids": [0],
        "engram_max_ngram_size": 3,
        "engram_vocab_size": 16,
        "engram_n_heads": 2,
        "engram_head_dim": 8,
        "engram_pad_token_id": 0,
        "engram_compressed_vocab_size": 4,
    }
    dst = "/tmp/dsv41_tiny_strategy"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.path.join(dst, "config.json"), "w") as handle:
        json.dump(cfg, handle)
    config = DeepseekV41Config(dst)
    assert "mtp" not in config.model_classes and config.vision is None
    model = Model.from_config(config)
    strategy, _bpw = create_q_strategy(model, None, config, 2.9, 6, 4, False)
    keys = set(strategy)
    assert any("wo_a.slice" in key for key in keys)
    assert any(key.endswith("engram.wkv") for key in keys)
    assert not any("weights_proj" in key for key in keys)
    assert not any(key.endswith(".ffn.gate") for key in keys)
    assert not any("engram.embed" in key for key in keys)
    assert strategy["head"] == 6
    return sorted(k for k in keys if "wo_a" in k or "engram" in k or k == "head")


def alias_rejected():
    """Renaming architectures to DeepseekV4ForCausalLM does not load a V4.1 config."""
    import json
    import os
    import shutil
    from exllamav3.architecture.deepseek_v4 import DeepseekV4Config

    src = json.load(open(os.path.join(_FIX, "deepseek_v41_flash_config.json")))
    src["architectures"] = ["DeepseekV4ForCausalLM"]
    dst = "/tmp/dsv41_alias_rename"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.path.join(dst, "config.json"), "w") as handle:
        json.dump(src, handle)
    try:
        DeepseekV4Config(dst)
    except ValueError as exc:
        rename_error = str(exc)
    else:
        raise AssertionError("aliased config should not construct")
    assert "hidden_size" in rename_error

    flat = json.loads(json.dumps(src))
    flat.update(flat["text_config"])
    dst2 = "/tmp/dsv41_alias_hoist"
    if os.path.isdir(dst2):
        shutil.rmtree(dst2)
    os.makedirs(dst2)
    with open(os.path.join(dst2, "config.json"), "w") as handle:
        json.dump(flat, handle)
    try:
        DeepseekV4Config(dst2)
    except KeyError as exc:
        hoist_error = exc.args[0]
    else:
        raise AssertionError("hoisted config should still fail on compress ratio 2")
    assert hoist_error == 2
    return {"rename": rename_error, "hoist_ratio": hoist_error}


def test_flash_module_keys_match_index():
    result = run_isolated(flash_keys, env = _STUB, timeout = 600)
    assert result["keys"] > 40000
    assert result["scales_only"] > 0


def test_tiny_strategy_skips_unquantized_tables():
    picked = run_isolated(tiny_strategy, env = _STUB, timeout = 300)
    assert any("wo_a.slice" in key for key in picked)
    assert any(key.endswith("engram.wkv") for key in picked)


def revision_0909():
    import json
    import os
    import shutil
    from exllamav3.architecture.deepseek_v41 import DeepseekV41Config
    from exllamav3.model.model import Model
    from exllamav3.modules.dsv41 import DSV41Attention

    cfg = {
        "architectures": ["DeepseekV41ForCausalLM"],
        "dsv41_revision": "0909",
        "vocab_size": 32,
        "hidden_size": 32,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "qk_rope_head_dim": 8,
        "q_lora_rank": 8,
        "o_lora_rank": 8,
        "o_groups": 2,
        "rms_norm_eps": 1e-6,
        "compress_ratios": [2],
        "rope_theta": 10000.0,
        "compress_rope_theta": 10000.0,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": 1,
        "sliding_window": 4,
        "kv_source_layer_ids": [0],
        "index_source_layer_ids": [0],
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 1,
        "candidate_source_layer_id": 0,
        "hc_mult": 2,
        "engram_layer_ids": [],
        "engram_num_embeddings": [],
    }
    dst = "/tmp/dsv41_rev_0909"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.path.join(dst, "config.json"), "w") as handle:
        json.dump(cfg, handle)
    config = DeepseekV41Config(dst)
    model = Model.from_config(config)
    attn = next(m for m in model if isinstance(m, DSV41Attention))
    assert config.dsv41_revision == "0909"
    assert attn.wo_a_act_quant is False
    assert attn.publish_index_k_always is False
    return config.dsv41_revision


def test_v4_alias_does_not_load_v41_config():
    result = run_isolated(alias_rejected, env = _STUB, timeout = 300)
    assert result["hoist_ratio"] == 2


def test_0909_revision_disables_wo_a_act_quant():
    assert run_isolated(revision_0909, env = _STUB, timeout = 300) == "0909"
