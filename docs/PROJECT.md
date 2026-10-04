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

## Pinned model contract

The converter and loader accept one upstream checkpoint:

- Repository: `paradigma-inc/limite-1b-violetto`
- Revision: `b1f3d572ccacb6919f4d64c321b70ba034ddaef2`
- File: `model.safetensors`
- Source SHA-256: `9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`

The production manifest is pinned to the immutable Hugging Face commit `f66f2e77b2bfd4e8aa80e0770943c3c0d3414f18` under `webgpu-q4-v2`. The loader validates the source identity, model constants, tensor inventory, shard sizes, and tensor ranges before using the artifact.

The model contract is 48 layers with hidden size 1,280, intermediate size 3,328, 10 query heads, 2 key/value heads, and head dimension 128. Local attention retains the current token plus 1,024 preceding tokens; layers 3, 7, …, 47 use global attention. RoPE covers the first 64 dimensions of each head with base 1,024. Value embeddings occur on layers 1, 4, …, 46, XSA is used on every layer, MUDD mixing occurs at layers 24 and 47, and the token embedding is tied to the vocabulary head. Logits use Paradigma's soft cap `23 * sigmoid((raw + 5) / 7.5)`.

The browser artifact uses symmetric `q4_block32` weights with FP16 scales and FP32 control tensors. Conversion fuses Q/K/V and gate/up matrices, folds the published projection scales, and stores the XSA coefficients after `tanh`, matching the reference computation.

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

Generation uses temperature 0.6, top-p 0.95, and top-k 50 on the GPU. Token ID 151643 is BOS, EOS, and PAD; `<|im_end|>` is not treated as EOS.

The runtime has a 4,096-token working context. The maximum output is the unused portion after the formatted prompt. Generation ends on EOS, when the context is full, or after a complete balanced `\boxed{...}` answer. If the model emits `<think>`, the box is accepted only after `</think>`; when the answer is inside `\[...\]` or `\(...\)`, the closing math delimiter is retained before stopping.

## Loading and performance

The first visit downloads about 556 MiB of model data. Successful shards are cached in OPFS, stale artifact shards are removed, and later visits load the pinned artifact locally. The browser still has to read the cached bytes, create GPU buffers, and compile pipelines on each page load. Private browsing, cleared site data, or insufficient storage quota can prevent persistence.

A valid local WebGPU run on the current test browser and hardware reached 56.78 generated tokens/second and returned the correct boxed answer. This is a spot measurement, not a portable benchmark; performance depends on the GPU, browser, thermals, prompt length, and generated sequence length.

## Deployment invariants

- Keep the app browser-only and compatible with static GitHub Pages hosting.
- Do not commit model weights to the application repository.
- Keep all production artifact URLs revision-pinned.
- Preserve `src/model.ts` as the readable JAX.js reference when changing the custom runtime.
- Verify both output correctness and inference in a real WebGPU browser after runtime changes.
