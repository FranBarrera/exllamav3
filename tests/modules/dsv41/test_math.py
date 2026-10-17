"""CPU checks of the V4.1 math against an independent transcription of the reference formulas."""
import importlib.util
import math
import os

import pytest
import torch

pytestmark = pytest.mark.nogpu

_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "exllamav3", "architecture", "dsv41_math.py",
)


def load_math():
    spec = importlib.util.spec_from_file_location("dsv41_math_under_test", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yarn_plain_and_scaled():
    math_mod = load_math()
    # Flash text rope: head pair-count 32 (qk_rope_head_dim 64), theta 10000, YaRN factor 16, context 65536
    dim, base = 64, 10000.0
    plain = math_mod.yarn_inv_freq(dim, base, 16.0, 0, 32.0, 1.0)
    # original_seq_len == 0 disables YaRN, so the factor must not change the table
    expected = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    assert torch.allclose(plain, expected)
    scaled = math_mod.yarn_inv_freq(dim, base, 16.0, 65536, 32.0, 1.0)
    # high-frequency end stays on the unscaled table; the low-frequency end is pulled down by the factor
    assert torch.allclose(scaled[0], expected[0])
    assert scaled[-1] < expected[-1]
    assert scaled[0] > scaled[-1] * 10


def test_rotary_hand_example_and_4d_broadcast():
    math_mod = load_math()
    x = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]])
    freqs = torch.polar(torch.ones(1, 2), torch.tensor([[0.0, math.pi / 2]]))
    y = math_mod.apply_rotary(x, freqs)
    # (1+0j)*e^0 = 1; (0+1j)*e^{i pi/2} = -1
    assert torch.allclose(y, torch.tensor([[[1.0, 0.0, -1.0, 0.0]]]), atol = 1e-5)
    wide = torch.zeros(1, 1, 2, 6)
    wide[..., -4:] = x.view(1, 1, 1, 4)
    out = math_mod.apply_rotary_tail(wide, freqs, 4)
    assert torch.allclose(out[0, 0, 0, -4:], y[0, 0], atol = 1e-5)
    assert torch.count_nonzero(out[..., :-4]) == 0


def test_fp8_fp4_properties():
    math_mod = load_math()
    zeros = math_mod.fp8_fake_quant(torch.zeros(2, 32), 32)
    assert torch.isfinite(zeros).all()
    exact = math_mod.fp8_fake_quant(torch.full((1, 32), 448.0), 32)
    assert torch.allclose(exact, torch.full((1, 32), 448.0))
    # last dim not divisible by the block: left unchanged (tiny configs)
    odd = torch.arange(6).float()
    assert torch.equal(math_mod.fp8_fake_quant(odd, 32), odd)
    level = math_mod.fp4_fake_quant(torch.full((1, 32), 6.0), 32, "e8m0")
    assert torch.allclose(level, torch.full((1, 32), 6.0))
    weight = torch.tensor([1.0, 0.0, 0.0, 0.0])
    scale = torch.tensor([127], dtype = torch.uint8)
    assert torch.allclose(math_mod.dequant_fp8_e8m0(weight, scale, 4), weight)


def test_sparse_attn_hand_and_invalid():
    math_mod = load_math()
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    idxs = torch.tensor([[[0]]], dtype = torch.int32)
    sink = torch.zeros(1)
    out = math_mod.sparse_attn(q, kv, sink, idxs, 1.0)
    # logits [q·k, sink] = [1, 0]; the output is that softmax weight times k
    weight = torch.exp(torch.tensor(1.0)) / (torch.exp(torch.tensor(1.0)) + 1.0)
    assert torch.allclose(out, torch.tensor([[[[weight, 0.0]]]]), atol = 1e-5)
    dead = math_mod.sparse_attn(q, kv, sink, torch.tensor([[[-1, -1]]], dtype = torch.int32), 1.0)
    assert torch.equal(dead, torch.zeros_like(q))
    empty = math_mod.sparse_attn(q, kv[:, :0], sink, torch.zeros(1, 1, 0, dtype = torch.int32), 1.0)
    assert torch.equal(empty, torch.zeros_like(q))


def test_candidate_blocks_pin_newest():
    math_mod = load_math()
    logits = torch.tensor([[1.0, 0.0, 0.0, 5.0]])
    mask = math_mod.select_candidate_blocks(logits, 4, 1, 2)
    # two blocks, amax 1 and 5; the newest block is pinned, so top-1 keeps positions 2 and 3
    assert torch.equal(mask, torch.tensor([[False, False, True, True]]))


def test_window_and_dspark_idxs():
    math_mod = load_math()
    prefill = math_mod.window_topk_idxs(4, 1, 3, 0)
    # query i sees slots 0..i and then -1 out to the window
    assert torch.equal(prefill[0, 0], torch.tensor([0, -1, -1], dtype = torch.int32))
    assert torch.equal(prefill[0, 2], torch.tensor([0, 1, 2], dtype = torch.int32))
    decode = math_mod.window_topk_idxs(4, 2, 1, 5)
    assert decode.shape == (2, 1, 4)
    draft = math_mod.dspark_topk_idxs(4, 2, 2, 3)
    expect = torch.tensor([0, 1, 2, 3, 4, 5], dtype = torch.int32)
    assert torch.equal(draft[0, 0], expect)
    assert torch.equal(draft[1, 1], expect)
    with pytest.raises(ValueError):
        math_mod.dspark_topk_idxs(4, 1, 2, 0)


def test_hc_schedule_does_not_mutate_residual():
    math_mod = load_math()
    mixes = torch.randn(1, 2, (2 + 2) * 2)
    scale = torch.tensor([0.2, 0.3, 0.4])
    base = torch.randn((2 + 2) * 2)
    pre, post, comb = math_mod.hc_split_sinkhorn(mixes, scale, base, 2, 3, 1e-6)
    assert pre.shape == (1, 2, 2) and comb.shape == (1, 2, 2, 2)
    assert torch.isfinite(comb).all()
    # column sums of a Sinkhorn matrix stay near 1 after the first normalization
    assert torch.allclose(comb.sum(dim = -2), torch.ones(1, 2, 2), atol = 1e-4)
    streams = torch.randn(1, 2, 2, 4)
    residual = streams.clone()
    out = math_mod.hc_post(torch.randn(1, 2, 4), residual, post, comb)
    assert torch.equal(residual, streams)
    assert out.shape == streams.shape
    ident = math_mod.identity_pre_mix(streams, 2)
    collapsed = math_mod.hc_pre(streams, ident)
    assert torch.allclose(collapsed, streams[:, :, 0, :])


def test_v4_immediate_pre_diverges_from_v41_delayed_pre():
    """Same weights. V4 collapses with this site's pre; V4.1 uses the previous site's pre."""
    math_mod = load_math()
    torch.manual_seed(0)
    b, s, h, d = 1, 2, 2, 4
    streams = torch.randn(b, s, h, d)
    mix_hc = (2 + h) * h
    fn = torch.randn(mix_hc, h * d)
    base = torch.randn(mix_hc)
    scale = torch.randn(3)
    eps = 1e-6
    iters = 4

    def norm_linear(x):
        flat = x.flatten(2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim = True) + eps)
        return torch.nn.functional.linear(flat, fn)

    def v4_site(x):
        mix = norm_linear(x)
        pre_w, post_w, comb_w = mix.split([h, h, h * h], dim = -1)
        pre_b, post_b, comb_b = base.split([h, h, h * h])
        pre_s, post_s, comb_s = scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = comb_w.view(*comb_w.shape[:-1], h, h) * comb_s + comb_b.view(h, h)
        comb = torch.softmax(comb, dim = -1) + eps
        comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(dim = -1, keepdim = True) + eps)
            comb = comb / (comb.sum(dim = -2, keepdim = True) + eps)
        collapsed = (pre.unsqueeze(-1) * x).sum(dim = 2)
        resid = post.unsqueeze(-1) * collapsed.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), x)
        return resid

    def v41_site(x, pre_in):
        pre, post, comb = math_mod.hc_split_sinkhorn(norm_linear(x), scale, base, h, iters, eps)
        y = math_mod.hc_pre(x, pre_in)
        return math_mod.hc_post(y, x, post, comb), pre

    out_v4 = v4_site(v4_site(streams))
    resid, pre = v41_site(streams, math_mod.identity_pre_mix(streams, h))
    out_v41, _pre = v41_site(resid, pre)
    diff = (out_v4 - out_v41).abs()
    assert float(diff.max()) > 1.0
    assert not torch.allclose(out_v4, out_v41, atol = 1e-3)


def test_pool_and_gate():
    math_mod = load_math()
    kv = torch.tensor([[[[1.0, 0.0], [3.0, 0.0]]]])  # [b, groups, ratio, d]
    score = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    pooled = math_mod.pool_groups(kv, score)
    assert torch.allclose(pooled, torch.tensor([[[2.0, 0.0]]]))
    streams = torch.randn(1, 3, 2, 4)
    key = torch.randn(1, 3, 2, 4)
    gate = math_mod.engram_gate(streams, key, torch.ones(2, 4), 1e-6)
    assert gate.shape == (1, 3, 2)
    assert torch.isfinite(gate).all()
    assert ((gate > 0) & (gate < 1)).all()


def test_engram_primes_are_not_reused_across_layers():
    math_mod = load_math()

    def next_prime(start, seen):
        candidate = start + 1
        while True:
            if candidate > 2 and candidate % 2 == 0:
                candidate += 1
                continue
            root = int(candidate ** 0.5)
            if candidate >= 2 and all(candidate % d for d in range(2, root + 1)) and candidate not in seen:
                return candidate
            candidate += 1 if candidate == 2 else 2

    primes = math_mod.engram_primes((1, 14), 3, 2, 30)
    seen = set()
    expect = next_prime(29, seen)
    assert primes[0][0][0] == expect
    seen.add(expect)
    # computing layer 14 alone, starting again at vocab_size, would hand it layer 1's prime
    assert primes[1][0][0] != expect
    flat = [p for layer in primes for row in layer for p in row]
    assert len(flat) == len(set(flat))


def test_hash_multipliers_and_xor():
    math_mod = load_math()
    import numpy as np
    vocab = 50
    bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
    generator = np.random.default_rng(10007 * 14)
    values = generator.integers(0, bound, size = 3, dtype = np.int64) * 2 + 1
    row = math_mod.hash_multipliers([14], 3, vocab)[0]
    assert torch.equal(row, torch.tensor(values, dtype = torch.int64))
    assert torch.all(row % 2 == 1)

    compressed = torch.tensor([[4, 5]])
    cache = torch.zeros(1, 2, dtype = torch.int64)
    primes = torch.tensor([[[7]]])  # one layer, one n-gram width, one head
    offsets = torch.zeros(1, 1, dtype = torch.int64)
    multipliers = torch.tensor([[3, 5]])
    hashes = math_mod.ngram_hash_ids(compressed, cache, 0, primes, offsets, multipliers, pad_id = 1)
    # position 1 looks back one token: (5*3) xor (4*5) = 15 xor 20 = 27; 27 % 7 = 6
    assert int(hashes[0, 1, 0, 0]) == ((5 * 3) ^ (4 * 5)) % 7
    assert torch.equal(cache[0], compressed[0])


def test_compressed_token_map_collapses_case():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    math_mod = load_math()
    tokenizer = Tokenizer(WordLevel({"A": 0, "a": 1, "B": 2}, unk_token = "[UNK]"))
    lookup, size = math_mod.build_compressed_token_map(tokenizer)
    assert lookup[0] == lookup[1]
    assert lookup[2] != lookup[0]
    assert size == 2
