# DeepSeek-V4.1-Flash gap analysis

Checkpoint: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`** (`architectures: ["DeepseekV41ForCausalLM"]`, `model_type: deepseek_v41`). The September checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash` at **`dba1be0a40aa45a94ad051997016db3960a90277`**. Comparison below is against this fork's `DeepseekV4ForCausalLM` and against those two official MIT trees (`inference/`, `encoding/`). Community EXL3 recipes were used only as a tensor-name compatibility target. Their code was not copied.

## 1017 versus 0909

`config.json`, `inference/config.json`, `tokenizer.json`, `model.safetensors.index.json` (96,085 tensors, 510,286,023,000 bytes), `inference/engram.py`, `inference/kernel.py`, `inference/vision.py`, `inference/convert.py`, `inference/generate.py`, and `inference/image_processor.py` are the same file at both revisions. Every one of the 48 weight shards has a new Git-LFS ETag and the same byte length. Shard 47 holds `layers.1.engram.embed.{weight,scale}` (101,535,150,936 bytes). Shard 48 holds `layers.14.engram.embed.{weight,scale}` (101,537,926,640 bytes). `gate.bias_vl` is not in those two shards: each trunk layer and each MTP block stores its own, so it arrives inside that layer's shard. Mixing a 0909 shard with a 1017 shard crosses checkpoint generations. Engram tables and every `gate.bias_vl` have to come from 1017.

`inference/model.py` has two forward-pass edits. Nothing else in that file changed.

1. `Attention.forward` and `DSparkAttention.forward` call `act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)` after the inverse RoPE and the view to `[batch, seq, groups, group_dim]`, before the grouped `wo_a` einsum. That is an in-place FP8 E4M3 round-trip with UE8M0 scales, block 32, on the per-group width. 0909 feeds the bf16/fp32 activations straight into `wo_a`.
2. `Indexer.forward` publishes `shared_attn.index_k = self.k_cache` whenever the layer owns keys. 0909 published only inside `if owns_k and latent is not None`, so a ratio-2 source whose group was still open (decode steps where `(start_pos + 1) % 2 != 0` on layers 2, 8, and 14) left the pointer on the previous owner's cache, which at the start of a step is layer 20's cache from the previous token.

Each KV source already had its own `k_cache` and `compress_kv_cache` in both revisions. The loader now keeps those caches under `ik.{layer}` and `ckv.{layer}` and publishes the shared pointer the way 1017 does. 0909 still publishes index keys only on the step that appends them.

`encoding/encoding.py` also changed, and it is not a forward pass: tool-result content may be a block list, a user message that is only tool results does not start a new turn for `drop_thinking`, and image-token validation runs before tool messages are merged. This tree's chat helper is still the short `<|User|>` / `<|Assistant|>` prompt. It does not implement `encoding.py`.

The two forwards are selected from the checkpoint because `config.json` cannot tell them apart. `dsv41_revision` (`"1017"` or `"0909"`, top level or under `text_config`) wins. Otherwise the loader reads `README.md`, then the directory path (`1017` / `1bae0e10` versus `0909` / `dba1be0a`). A directory with none of those defaults to 1017.

## Would an architecture alias be enough?

**No.** Registering `DeepseekV41ForCausalLM` as the existing `DeepseekV4` class, or rewriting `architectures` in `config.json` to `DeepseekV4ForCausalLM`, does not load this checkpoint. Remapping the few config fields V4 knows about still leaves the loader asking for thousands of tensors that V4.1 does not have, and ignoring tensors V4.1 does have. The attention algorithm and the mHC schedule are different, so a loaded graph would not compute the same function.

Measured on the Flash-1017 `config.json` and on the collapsed `model.safetensors.index.json` (96,085 tensors, 510,286,023,000 bytes, 114 name patterns). The V4 loader was the real `DeepseekV4Config` plus `Model.from_config`, not a hand-written key list. `.scale` sidecars were excluded from the V4.1 side because V4 linears load them optionally and do not declare them (48,496 core names remain).

| Attempt | Result |
| --- | --- |
| Rename `architectures` to `DeepseekV4ForCausalLM` only | `ValueError: Missing any of the following keys: ['hidden_size']`. V4 reads `hidden_size` at the top of `config.json`. On this checkpoint it lives under `text_config`. |
| Also hoist `text_config` onto the top level | `KeyError: 2`. V4's ratio table is only `{0: sliding, 4: csa, 128: hca}`. V4.1 ratios are 0, 1, and 2. |
| Hoist, rewrite `2 → 4` and `1 → 128`, and copy `vision_config` into V4's flat `vision_n_layers` / `vision_dim` / … keys. Sentinel tensors so MTP and vision register. | Config constructs. `layer_types` starts `sliding, sliding, csa, …` (40 trunk layers). `num_hash_layers` stays at the V4 default 3. MTP 3 and vision turn on. The loader then asks for **50,953** keys. **2,558** of those names are absent from the V4.1 index. **101** V4.1 core names are never read. |

Pattern counts that still disagree after that rewrite (`v4` vs `v41`):

| Pattern | V4 loader | V4.1 index |
| --- | ---: | ---: |
| `layers.N.attn.compressor.ape` | 38 | 0 |
| `layers.N.attn.compressor.norm.weight` / `wkv.weight` | 38 | 4 |
| `layers.N.attn.compressor.wgate.weight` | 38 | 3 |
| `layers.N.attn.indexer.compressor.{ape,norm,wgate,wkv}` | 18 each | 0 |
| `layers.N.attn.indexer.wq_b.weight` / `weights_proj.weight` | 18 | 8 |
| `layers.N.attn.indexer.wk.weight` / `k_norm.weight` | 0 | 4 |
| `layers.N.engram.{embed.weight,q_weight,k_weight,wkv.weight}` | 0 | 2 each |
| `layers.N.ffn.gate.tid2eid` | 3 | 0 |
| `hc_head_{fn,base,scale}` | 1 each | 0 |
| `mtp.N.ffn.experts.E.w{1,2,3}.weight` | 1152 | 384 |
| `mtp.N.hc_head_{fn,base,scale}` | 1 each | 0 |
| `mtp.N.markov_head.markov_w{1,2}.weight` | 1 | 0 |
| `mtp.N.markov_head.{embed,head}.weight` | 0 | 1 |
| `mtp.N.ffn.gate.bias_vl` | 0 | 3 |
| `image_{start,end,newline}`, aligner biases, vision `wqkv`/`wo`/`patch` biases | 0 | present |

A forward of one V4 `TransformerBlock` on this machine raises `RuntimeError: exllamav3_ext.hc_mix_num_chunks is unavailable` (no CUDA; the extension is stubbed). The official V4.1 `inference/model.py` does not import either: `ModuleNotFoundError: No module named 'image_processor'`. `image_processor.py` is not in the published inference tree, so its numeric constants were not guessed.

The schedule difference can be measured without those imports. Same random weights, same input, two sites (attention then FFN):

- V4's torch fallback (`HyperConnection._mix_unfused`) collapses the residual with **this site's** `pre` immediately: `collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)`.
- V4.1 computes `pre` here and consumes it at the **next** sublayer. The first site starts from a one-hot mix on stream 0.

Max absolute difference **2.992461**, mean **1.316302**. An alias that only renamed modules would still produce a different block output. `tests/modules/dsv41/test_math.py::test_v4_immediate_pre_diverges_from_v41_delayed_pre` locks that split. `tests/modules/dsv41/test_graph.py::test_v4_alias_does_not_load_v41_config` locks the `hidden_size` and ratio-`2` failures.

Small patches on the V4 class (nested config, extra ratio entries, optional missing keys) would still leave CSA2, the delayed mHC, Engram, the DSpark expert width, and the vision bias set unimplemented. V4.1 is a separate architecture: `DeepseekV41ForCausalLM`.

## Config

V4.1 is nested: `text_config`, `vision_config`, and a top-level `quantization_config` (fp8 weights, expert fp4, `weight_block_size [32, 32]`, `scale_fmt ue8m0`). The reader accepts that nesting and a flat tiny config (`text_config->key` or `key`).

Text shape: hidden 5120, 40 layers, 64 heads, 1 KV head, head dim 512, RoPE dim 64, `q_lora_rank` 1280, `o_lora_rank` 1024, `o_groups` 8, vocab 129280. MoE: intermediate 2304, 384 routed / 6 active / 1 shared, `sqrtsoftplus`, `noaux_tc`, `norm_topk_prob`, routed scale 1.5, SwiGLU clamp 10, RMS eps `1e-20`. RoPE: theta 10000, compress theta 160000, YaRN factor 16, `beta_fast` 32, `beta_slow` 1, original context 65536, max position 1048576. Window 128. Indexer: 32 heads, head dim 128, top-k 512. mHC: mult 4, 20 Sinkhorn iterations, eps `1e-6`.

`compress_ratios` has 43 entries: 40 trunk layers plus 3 MTP. The trunk is `[0, 0] + [2] * 18 + [1] * 20`. Ratio 0 is a sliding window only. Ratio `r > 0` attends to a KV compressed `r`-to-1. One compressed KV and one indexer are **owned by the first source layer of a ratio group** and then shared: `kv_source_layer_ids` is `[2, 8, 14, 20]`. `wgate` exists only when `ratio > 1`, so layers 2, 8, and 14 (3 tensors). Layer 20 is ratio 1 (`wkv` + norm, no gate). V4 instead builds a compressor, an absolute-position embedding, and (on CSA) a second indexer compressor on every compressed layer.

Index sources are `[2, 8, 14, 20, 24, 28, 32, 36]`. The candidate mask is published by layer 20 (`candidate_topk_blocks` 2048, `candidate_block_size` 8). Later indexers reuse it. Non-index layers reuse the published top-k.

## Attention (CSA2)

Do not route this through `DSV4Attention` / `DSV4Compressor`.

- Latent Q: `wq_a`, `q_norm`, `wq_b`. Shared KV: `wkv`, `kv_norm`.
- Grouped `wo_a` is block-diagonal over `o_groups` (8), then `wo_b`. Converted slices are `layers.N.attn.wo_a.slice.{g}`, matching the existing grouped-`wo_a` layout.
- Learned `attn_sink` per head.
- Inverse RoPE (adjacent pairs) on the attention output. `softmax_scale = head_dim ** -0.5`.
- YaRN runs only when `compress_ratio > 0`, on `compress_rope_theta`. Sliding layers use plain `rope_theta` and `original_seq_len = 0`. The indexer uses the same RoPE table as its attention layer. The latent position is `j * ratio` (first token of the group). The indexer consumes the pre-RoPE latent; RoPE is applied after.
- Window KV is fake-quantized FP8, block 32, E8M0. Compressed KV is fake-quantized FP4, block 16, E4M3 scales. Indexer Q/K are fake-quantized FP4, block 32, E8M0. The torch fake-quant follows the reference formulas. It is not bit-identical to the TileLang kernels. A last dimension that is not divisible by the block is left unchanged so tiny configs still run.

Indexer projections: `wq_b` from the Q latent (has `.scale`), `weights_proj` (no scale, left unquantized), and `wk` + `k_norm` only on KV owners (no scale on `wk`). Scores are `relu(q · k) * weights`, summed over heads. The weights multiplier is `softmax_scale * n_heads ** -0.5` with `softmax_scale = index_head_dim ** -0.5`. Top-k is re-sorted into position order; unreachable positions become -1, then the window length is added as an offset.

Shared state is one slot: compressed KV and index K from the latest KV source, top-k from the latest index source, candidates from layer 20. Layers must run in order. During conversion that bag is `quant_preserve["dsv41"]` (CPU). During generation it is `DSV41State`.

## mHC

V4 `HyperConnection.mix` applies `pre` inside the mix. V4.1 stores `pre` for the next sublayer. Attention uses the previous FFN's `pre`. The FFN uses this attention's `pre`. The final logits use the last FFN's `pre` via `hc_pre`, then RMSNorm, then the head. There is no `hc_head_*` tensor and no `HyperHead`.

Sinkhorn matches `kernel.py`: sigmoid `pre` plus eps, `2 * sigmoid` post, softmax plus eps, then column/row Sinkhorn. `hc_post` does not mutate the residual in place. The mix RMS is over the flattened `hc * d` stream with `rms_norm_eps` (`1e-20`), then `linear(x) * rsqrt`. Weights still load through `HyperConnection` (`{key}_fn`, `_base`, `_scale`). The delayed multiply is pure torch. `HyperConnection.mix` / `mix_norm` are not called.

The initial `pre` is one-hot on stream 0.

## Engram

Layers 1 and 14. Applied to the hyper-connection residual **before** the block and before any DSpark tap. Engram is a child of the block (`layers.N.engram`), not a top-level module, because the model layer map requires consecutive unique `layer_idx` values on top-level modules.

Tables: `engram_num_embeddings` `[384006168, 384016682]`, head dim 256, 8 heads, max n-gram 4, prime-search start 16,000,000, pad id 2, compressed vocab 99,092. The row width is the head dim 256, not `n_heads * head_dim`. Each FP8 table is about 91.6 GiB. E8M0 scales (one byte per 32 elements along the row) add about 2.9 GiB each, so the two tables plus scales are about 189 GiB.

The hash is the reference one: compressed token map (NFKC, NFD, strip accents, lower case, whitespace collapse), odd int64 multipliers from `np.random.default_rng(10007 * layer_id)`, XOR-folded n-grams, prime buckets, exclusive-prefix offsets. Primes are drawn once across all Engram layers and never reused. `DEAD = -1` stops an n-gram at an image span (`token_types >= 0` is image; text is -1).

The gate is a per-(token, hc copy) normalized dot, signed square root, then sigmoid. `q_weight * k_weight` is the only use of those parameters. `wkv` maps `24 * 256 = 6144` inputs to `hidden * (hc_mult + 1) = 25600`. That projection is quantized. `q_weight` and `k_weight` stay raw. `embed.weight` and `embed.scale` are not in `get_tensors()`; they are additional compiled tensors streamed from the source shard. `weights_numel` is only `wkv` (`24 * 256 * 5120 * 5`), so the reported bits-per-weight does not treat the tables as quantized weights. `--ngram_bits` / `NGramEmbedding` does not apply.

`build_compressed_token_map` asserts the compressed size against `engram_compressed_vocab_size` when the real tokenizer is loaded. The unit test covers the collapse with a WordLevel tokenizer (`A` and `a` share an id). It has not been run on the Hugging Face DeepSeek tokenizer.

## MoE

`BlockSparseMLP` with `router_type="sqrtsp"`. Not `sqrtsp_hash`. Score bias is `gate.bias`. Vision-language bias is `gate.bias_vl` and is loaded and kept (some published EXL3 trees drop it). The existing router selects with `indexed_embeddings`, not `token_types`, so the image mask is not rewired onto `bias_vl`. Experts are `experts.{expert_idx}.w1/w3/w2`. There is no `tid2eid` table. SwiGLU uses `act_limit=swiglu_limit`. `gate.weight` is unquantized. DSpark experts are a different width: 128 routed, 3 active. MTP also has `gate.bias_vl`.

## DSpark / MTP

Three MTP blocks, compress ratio 0, block size 5, noise token 128799, tap layers `[37, 38, 39]`, Markov rank 256. A tap is the stream mean of the residual **entering** that layer (after Engram if that layer had one; 37–39 do not). Prefill (`start_pos == 0`) only seeds the window KV from the main hidden state and returns it unchanged. Decode attends over the window ring plus the draft block's own KV.

Entry is `mtp.0.main_proj` (input `hidden * n_taps`) plus `main_norm`. Exit is the last block's `hc_pre`, then `mtp.N.norm`, then the **shared** trunk head. Embeddings and the head are not duplicated. Markov tensors are `mtp.N.markov_head.embed` and `mtp.N.markov_head.head` (V4 uses `markov_w1` / `markov_w2`), unquantized. Confidence is `mtp.N.confidence_head.proj` (input `hidden + markov_rank`), unquantized. V4 MTP uses the trunk expert count (384) and `mtp.N.hc_head_*`.

Side-model quantization is uncalibrated. Default `mtp_bits` is 4. Speculative decoding is not wired: `prepare_inputs` raises `NotImplementedError` if it is given 2D token ids. The weights still convert.

## Vision

A separate tower from V4: 32 layers, dim 1024, 16 heads, intermediate 2816, patch 14, 2D RoPE on half the head dim, fused `wqkv` plus bias, `wo` plus bias, MLP `w1` as fused gate|up (no bias) plus `w2` (SiLU), GELU aligner `w1`/`w2` with bias, downsample 3. Learned `image_start` / `image_end` / `image_newline`. There is no `image_pad`. `image_token_id` is 129264. Vision weights have no `.scale`. RMS eps is `1e-6`.

`default_vision_bits` is left unset, so convert copies the tower raw unless `--vision_bits` is 1–8. The generator multimodal path is not wired (`prepare_inputs` raises). V4 looks for a top-level `vision_n_layers` and for `image_pad`, and it does not require the vision biases.

## Quantization and checkpoint layout

Source tensors **with** `.scale` (fp8 or fp4): attention `wq_a` / `wq_b` / `wkv` / `wo_a` / `wo_b`, indexer `wq_b`, expert and shared-expert `w1` / `w2` / `w3`, Engram `embed` and `wkv`, the same on MTP, and `mtp.0.main_proj`.

Source tensors **without** `.scale`: token embed, head, norms, `attn_sink`, `hc_*`, `gate.weight` / `gate.bias` / `gate.bias_vl`, compressor `wkv` / `wgate`, indexer `wk` / `weights_proj` / `k_norm`, Engram `q_weight` / `k_weight`, the whole vision tower and aligner, the image tokens, Markov embed/head, confidence, `main_norm`. `Linear.load_fp16` already dequantizes DeepSeek `.scale` E8M0 grids.

`wo_a` uses `select_hq_bits=2` (attention `q_priority` 4, so children inherit it). Shared experts are `GatedMLP` with `select_hq_bits=2`; `--hq` still floors them. `weights_proj` and `gate.weight` use `qmap=None`. Reported module bpw is bits over `weights_numel`. Engram contributes only `wkv`. The file is larger because the tables are streamed raw.

Calibration forwards one top-level module at a time. Cross-module state survives only inside `params["quant_preserve"]`. The V4.1 bag (shared KV, delayed `pre`, taps, Engram cache) is written there, moved to CPU, and checkpointed as `quant_preserve.safetensors` next to `job.json` when `caps["checkpoint_quant_preserve"]` is set. Resuming an older checkpoint that lacks `quant_preserve_saved` raises rather than quantizing with an empty cache. Other architectures do not set the cap, so their checkpoints stay as they were.

Engram tables at or above 256 MiB are stream-copied into their own safetensors shard, preserving the source dtype string (`F8_E4M3`, `F8_E8M0`). They are not loaded into the conversion process.

Published EXL3 name compatibility (the Mia vLLM TP=2 tree): `layers.N.attn.wo_a.slice.G.{trellis,suh,svh,mul1}`, compressor `wkv` (4) and `wgate` (3) quantized, `indexer.weights_proj` and `gate.weight` as raw `.weight`, experts `ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}`. This fork **keeps** `gate.bias_vl` and the Engram `embed.weight` / `embed.scale` passthrough. A loader that rejects unknown tensors has to ignore those.

`supports_tp` is false. `guaranteed_rollback` is 0. `caps["kv_cache"]` and `caps["recurrent_cache"]` are not set. There is no paged 1M cache.

## What was verified here

CPU, no CUDA. `EXLLAMA_EXT_STUB=1`. `pytest tests/modules/dsv41`: 22 passed.

- Math against an independent transcription of the reference formulas: YaRN on/off, rotary (including the 4D broadcast), FP8/FP4 fake-quant properties, the 1017 `wo_a` FP8 round-trip versus an identity 0909 input, the index-key publish rule, sparse attention, candidate-block pin, window and DSpark indices, Sinkhorn, Engram primes not reused across layers, hash multipliers and XOR order, compressed-token case collapse.
- Revision selection: `dsv41_revision`, README title, and directory name. A bare directory defaults to 1017. A config pinned to 0909 builds attention with `wo_a_act_quant` and `publish_index_k_always` off, including on the DSpark attention of a 1017-default Flash tree (those flags stay on).
- `source_keys` equals the Flash-1017 index patterns (96,085 names, 114 patterns), including 4 compressor `wkv`, 3 `wgate`, no `hc_head_fn`, no `tid2eid`.
- Full Flash module tree (text + MTP + vision, sentinel weights) walks to the index except `.scale` sidecars. `wo_a` is 40 × 8 slices. An index-source attention has `q_priority == 4`. Engram `weights_numel` is `wkv` only.
- Tiny flat config: quant strategy includes `wo_a.slice` and `engram.wkv`, and excludes `weights_proj`, `ffn.gate`, and `engram.embed`. Head bits stay 6.
- Stream-copy roundtrip keeps `F8_E4M3` bytes. `quant_preserve` roundtrips a dotted key. An empty bag does not write a file.
- Alias failures and the mHC numeric split above.

## What a full-model run still has to confirm

- Numeric match of a real forward against the TileLang kernels. Fake-quant is not bit-exact, and the reference package does not import in this environment.
- An actual `convert.py` of the 510 GB checkpoint, including peak memory once a layer of dequantized experts and per-expert Hessians are live.
- `engram_compressed_vocab_size == 99092` on the real DeepSeek tokenizer.
- Generator behavior: vision prefill, DSpark speculative decode, and VL routing through `token_types` rather than `indexed_embeddings`.
- Tensor-parallel load and a paged 1M KV cache. Neither is implemented.
