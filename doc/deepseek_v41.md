# DeepSeek-V4.1-Flash (EXL3)

`DeepseekV41ForCausalLM` converts [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision **`1bae0e1088993b9b82de08f59de5be5a34e42170`**. Pin that revision when you download. `config.json` is byte-identical to [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (`dba1be0a40aa45a94ad051997016db3960a90277`, the 0909 checkpoint), and the tensor catalog is the same 96,085 names. The weights are not: all 48 shards have new bytes. This is not an alias of `DeepseekV4ForCausalLM`. See [deepseek_v41_gap_analysis.md](deepseek_v41_gap_analysis.md).

Use the 1017 tree for every shard. `model-00047-of-00048.safetensors` (101,535,150,936 bytes) is `layers.1.engram.embed.weight` and `.scale`. `model-00048-of-00048.safetensors` (101,537,926,640 bytes) is `layers.14.engram.embed.weight` and `.scale`. `gate.bias_vl` sits in each layer's own shard (trunk and MTP), not in 47/48, and those shards changed too. A 0909 Engram table or a 0909 `gate.bias_vl` next to 1017 weights is a cross-generation checkpoint.

1017's `inference/model.py` adds an FP8 E4M3 / UE8M0 round-trip (`act_quant`, block 32) on the grouped input of `wo_a` in trunk attention and in DSpark attention, and it republishes an index-key owner's cache when the compression group is still open. Both are on for a 1017 directory. Set `"dsv41_revision": "0909"` in `config.json` (top level or `text_config`) to keep the September forward. The loader also reads `README.md` and the directory name when that key is absent. A directory with neither hint defaults to 1017.

## Convert

`--work_dir` is required. Leave `--vision_bits` unset so the vision tower is copied raw. Do not pass `--ngram_bits`: the Engram tables are not `NGramEmbedding` tables, and they are streamed through unquantized.

```
python convert.py \
  -i /path/to/DeepSeek-V4.1-Flash-1017 \
  -o /path/to/DeepSeek-V4.1-Flash-1017-EXL3 \
  -w /path/to/dsv41-1017-work \
  -b 2.9 -hb 6 -mb 4 \
  -cr 250 -cc 2048 \
  -cb mul1 -ss 8192
```

That command is the 2.9 bpw mul1 budget (`-b 2.9 -hb 6 -mb 4 -cb mul1`) aimed at the [TensorFold TP2 kit](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) for two GB10s. The kit's precompiled kernels were built for the Mia-AiLab EXL3 2.9 bpw pack (codebook **mul1**), and they dispatch on the **per-tensor** width, not on the average. Match these widths or the cubins do not line up with the tensors:

| Tensor | Bits |
| --- | ---: |
| routed experts | 3, except layers 18–22 at 2 |
| shared experts | 5 on layers 0–10 and 30–39, 4 on 11–29 |
| attention | 5 |
| `lm_head` | 6 |
| MTP / DSpark | 4 |
| indexer `wk` | 8 |
| Engram `wkv` | 5 on layer 1, 4 on layer 14 |
| embeddings | 16 (bf16) |
| vision tower | 16 (copied raw) |

`-b 2.9` spends a budget; it does not emit that table by itself. `--hq` only lifts attention (`q_priority` 4, so the floor is 4 with `select_hq_bits` 2) and the shared experts. To hit the kit's map, pass `-rcp` with those widths (mul1, so 2 and 3 and 4 and 5 are legal). `out_scales` for that pack is `always`. Engram hash tables stay raw, from the 1017 shards 47 and 48.

Resume with `python convert.py -r -w /path/to/dsv41-1017-work`. A checkpoint written by this architecture stores the shared KV, the delayed mHC pre-mix, the DSpark taps, and the Engram token cache in `ckpt/quant_preserve.safetensors`. Resuming a job directory from before that file existed stops with an error instead of quantizing against an empty cache. Start that job again.

## Calibration

The default corpus is the bundled mix (`exllamav3/conversion/standard_cal_data`). To use a self-sampled trace instead, build one with a model that can run prefill and pass it as `-cd`:

```
python sc_trace.py -m /path/to/a-runnable-model -o cal_trace.safetensors
python convert.py ... -cd cal_trace.safetensors
```

`sc_trace.py` needs a loaded model. It cannot be pointed at the 510 GB checkpoint on a machine that cannot run it. MTP layers are quantized without a calibration forward (`--mtp_bits`, default 4).

## Disk and memory

The source index is 96,085 tensors and 510,286,023,000 bytes (about 475 GiB).

Engram (layers 1 and 14) is two FP8 tables of shape `[384006168, 256]` and `[384016682, 256]`, about 91.6 GiB each, plus E8M0 scales of about 2.9 GiB each. Together that is about 189 GiB. Those tensors are stream-copied into their own output shards and are not resident during conversion. `wkv` on each Engram is quantized. The reported bits-per-weight does not count the raw tables.

One trunk layer of routed experts, dequantized to fp16, is `384 * 3 * 5120 * 2304 * 2` bytes, about 25.3 GiB (27.2 GB). Down-projection Hessians are per expert (`2304²` fp32), about 8 GiB for the layer. Gate and up share one input Hessian. A 128 GB GPU (or one DGX Spark's unified memory) has room for that plus the rest of the layer. An 80 GB GPU is tight. Conversion can run on one Spark or on an H200/B200. The output names match the usual EXL3 layout (`wo_a.slice.{g}`, `ffn.experts.{e}.w{1,2,3}`), which is what a TP=2 vLLM load of this layout expects. This fork does not implement tensor parallel itself.

The output is the quantized weights (about 2.9 bpw on the budgeted tensors, 6 on the head, 4 on MTP) plus the raw Engram tables, the raw vision tower, and the unquantized embeddings. It is not a 2.9-bit packing of the whole 475 GiB source.

## What the loader keeps that some EXL3 trees drop

- `gate.bias_vl` on trunk and MTP routers.
- Engram `embed.weight` and `embed.scale` as raw FP8.

`indexer.weights_proj` and `gate.weight` stay unquantized `.weight`. A downstream loader that rejects unknown keys has to skip `bias_vl` and the Engram tables.

## Limitations

Verified on CPU with `EXLLAMA_EXT_STUB=1` (`pytest tests/modules/dsv41`): math against the reference formulas, including the 1017 `wo_a` FP8 round-trip and the index-key publish rule, revision selection, the Flash-1017 tensor-name map, the quant strategy on a tiny config, stream copy, and the quant-preserve bag. See the gap analysis for the alias experiment and the 0909/1017 diff.

Not verified, because this environment has no CUDA and cannot load the checkpoint:

- A real `convert.py` run and the resulting bpw.
- Peak memory once Hessians and dequantized experts are live.
- Forward numerics against the official TileLang kernels. The torch fake-quant follows `kernel.py`'s `act_quant` formula and is not bit-identical to those kernels. This environment does not run the TileLang reference.
- The compressed-vocab assert (`99092`) against the real DeepSeek tokenizer. The unit test uses a WordLevel tokenizer only.
- Generator behavior. Vision prefill and DSpark speculative decode raise `NotImplementedError`. `bias_vl` is stored, but routing still uses `indexed_embeddings` rather than `token_types`.
- Paged 1M KV cache, tensor parallel, and in-place speculative rewind (`guaranteed_rollback` is 0).
