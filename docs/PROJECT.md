# Kalkulator project brief

## Goal

Run Paradigma's Violetto Limite 1B mathematics model entirely in a WebGPU browser. The public app is static: prompts and inference stay on the user's device, with no inference server or account.

## Production runtime

- Source: <https://github.com/gustofied/kalkulator>
- Live: <https://gustofied.github.io/kalkulator/>
- App: static Vite/TypeScript, deployed to GitHub Pages by `.github/workflows/pages.yml`
- Execution: one custom Q4 WebGPU runtime in a dedicated Web Worker
- Artifact: six immutable binary shards, about 556 MiB total
- Cache: each complete shard is stored in OPFS; a warm load reads the local copy instead of downloading it again
- Interface: editorial black/violet UI with GSAP motion and KaTeX rendering

There is no production JAX.js path, runtime probe, A/B switch, or fallback engine. [`src/model.ts`](../src/model.ts) remains the JAX.js reference implementation and architecture oracle.

[`src/limite-config.ts`](../src/limite-config.ts) is the single serving policy: pinned Q4 artifact and tokenizer, context, sampling, batch size, and time limits. It has no URL parameters, model selector, or runtime overrides. JAX.js is a development dependency for the reference; only its loader package's tokenizer utility is used by the app.

## Pinned model contract

The converter and loader accept one upstream checkpoint:

- Repository: `paradigma-inc/limite-1b-violetto`
- Revision: `b1f3d572ccacb6919f4d64c321b70ba034ddaef2`
- File: `model.safetensors`
- Source SHA-256: `9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`

The production manifest is pinned to the immutable Hugging Face commit `f66f2e77b2bfd4e8aa80e0770943c3c0d3414f18` under `webgpu-q4-v2`. The loader validates the source identity, model constants, tensor inventory, shard sizes, and tensor ranges before using the artifact.

The model contract is 48 layers with hidden size 1,280, intermediate size 3,328, 10 query heads, 2 key/value heads, and head dimension 128. Local attention retains the current token plus 1,024 preceding tokens; layers 3, 7, …, 47 use global attention. RoPE covers the first 64 dimensions of each head with base 1,024. Value embeddings occur on layers 1, 4, …, 46, XSA is used on every layer, MUDD mixing occurs at layers 24 and 47, and the token embedding is tied to the vocabulary head. Logits use Paradigma's soft cap `23 * sigmoid((raw + 5) / 7.5)`.

The browser artifact uses symmetric `q4_block32` weights with FP16 scales and FP32 control tensors. Conversion fuses Q/K/V and gate/up matrices, folds the published projection scales, and stores the XSA coefficients after `tanh`, matching the reference computation.

`scripts/pack-limite-q4.py` is the offline reproducible packer for this artifact, not part of deployment or inference. It requires explicit `--input` and `--output` paths and refuses to overwrite an existing output directory.

## Prompt and generation contract

Input is NFC-normalized and formatted exactly as:

```text
<|endoftext|><|im_start|>system
You are a helpful assistant.
Please reason step by step, and put your final answer within \boxed{}.<|im_end|>
<|im_start|>user
{problem}<|im_end|>
<|im_start|>assistant
```

Generation uses temperature 0.6 and full-vocabulary top-p 0.95 on the GPU, without a top-k cutoff. Only the 151,667 tokenizer IDs are eligible; the 13 padded model rows are excluded. `src/limite-engine/sampler.ts` contains the single production sampler. Token ID 151643 is BOS, EOS, and PAD; `<|im_end|>` is not treated as EOS.

The runtime supports 131,072 context tokens and at most 126,976 output tokens, bounded by the space remaining after the prompt. These budgets follow Paradigma's technical report. Generation ends on the model's EOS, the token budget, or cancellation. A boxed expression is not a stop token: intermediate boxes must not truncate a solution. There is no wall-clock solve cutoff, automatic retry, or answer-forcing pass. A 150-second inactivity watchdog recovers a stalled worker; status and token progress reset it, and returning from a background tab grants a fresh interval.

All working text stays in the six-line scrolling area. After EOS, the last complete boxed answer is printed beneath it. An unfinished thinking block or token-limited output is not presented as a final answer. Completion flushes the visual writer, and the interface exposes preparation progress, elapsed solve time, and an explicit limit or error state.

## Loading and performance

The first visit downloads about 556 MiB of model data. Successful shards are cached in OPFS, stale artifact shards are removed, and later visits load the pinned artifact locally. Every downloaded or cached shard is checked against its manifest SHA-256; damaged cache entries are replaced. The browser still has to read the cached bytes, create GPU buffers, and compile pipelines on each page load. Private browsing, cleared site data, or insufficient storage quota can prevent persistence.

Keys and values are stored as packed FP16 pairs; queries, reductions, softmax, and activations remain FP32. Global caches start at 4K tokens and grow geometrically while preserving the prefix. Local caches retain their fixed 1,025-slot rings. A new short solve releases an oversized global cache. Attention-cache storage is about 84 MiB initially, 228 MiB at 16K, and 1,572 MiB at 128K, excluding weights, scratch, and temporary allocations during growth. The maximum is a supported context budget, not a guarantee that every device has enough memory.

Attention scores 32 keys in parallel per tile and merges four tiles per 128-key partition. RoPE factors are calculated only for the current decode batch, rather than allocating a full-context table. These are the only production kernels; no runtime benchmark or kernel selection is performed.

See the dated measurements in RESULTS.md. Performance depends on the GPU, browser, thermals, prompt length, and generated sequence length. Longer generation budgets permit longer solutions; they do not establish answer accuracy or shorter waiting times.

See [RESULTS.md](RESULTS.md) for dated checks and unresolved answer-quality limits. A working browser runtime is not evidence that all Paradigma blog problems are solved reliably.

Reference: [Limite 1B Violetto technical report](https://paradigma.inc/research/limite-1b-violetto.pdf), Tables 5–6 and Section 5.

## Deployment invariants

- Keep the app browser-only and compatible with static GitHub Pages hosting.
- Do not commit model weights to the application repository.
- Keep all production artifact URLs revision-pinned.
- Preserve `src/model.ts` as the readable JAX.js reference when changing the custom runtime.
- Verify both output correctness and inference in a real WebGPU browser after runtime changes.
