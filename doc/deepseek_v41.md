# DeepSeek-V4.1-Flash (EXL3)

`DeepseekV41ForCausalLM` converts [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017). The same `config.json` is shared with DeepSeek-V4.1-Flash. This is not an alias of `DeepseekV4ForCausalLM`. See [deepseek_v41_gap_analysis.md](deepseek_v41_gap_analysis.md) for the measured config and tensor mismatch.

## Convert

`--work_dir` is required. Leave `--vision_bits` unset so the vision tower is copied raw. Do not pass `--ngram_bits`: the Engram tables are not `NGramEmbedding` tables, and they are streamed through unquantized.

```
python convert.py \
  -i /path/to/DeepSeek-V4.1-Flash-1017 \
  -o /path/to/DeepSeek-V4.1-Flash-EXL3 \
  -w /path/to/dsv41-work \
  -b 2.9 -hb 6 -mb 4 \
  -cr 250 -cc 2048 \
  -cb mul1 -ss 8192
```

`-hq` raises the floor on attention (`q_priority` 4) and on the shared experts. The routed experts stay on the `--bits` budget.

Resume with `python convert.py -r -w /path/to/dsv41-work`. A checkpoint written by this architecture stores the shared KV, the delayed mHC pre-mix, the DSpark taps, and the Engram token cache in `ckpt/quant_preserve.safetensors`. Resuming a job directory from before that file existed stops with an error instead of quantizing against an empty cache. Start that job again.

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

Verified on CPU with `EXLLAMA_EXT_STUB=1` (`pytest tests/modules/dsv41`, 18 passed): math against the reference formulas, the Flash-1017 tensor-name map, the quant strategy on a tiny config, stream copy, and the quant-preserve bag. See the gap analysis for the alias experiment.

Not verified, because this environment has no CUDA and cannot load the checkpoint:

- A real `convert.py` run and the resulting bpw.
- Peak memory once Hessians and dequantized experts are live.
- Forward numerics against the official TileLang kernels. The torch fake-quant matches the published formulas and is not bit-identical to those kernels. `inference/model.py` does not import here (`image_processor` is absent).
- The compressed-vocab assert (`99092`) against the real DeepSeek tokenizer. The unit test uses a WordLevel tokenizer only.
- Generator behavior. Vision prefill and DSpark speculative decode raise `NotImplementedError`. `bias_vl` is stored, but routing still uses `indexed_embeddings` rather than `token_types`.
- Paged 1M KV cache, tensor parallel, and in-place speculative rewind (`guaranteed_rollback` is 0).
