# Kalkulator project brief

## Goal

Run Paradigma's Violetto Limite 1B mathematics model entirely in a WebGPU browser. The public app is static: prompts and inference stay on the user's device, with no inference server or account.

## Production runtime

- Source: <https://github.com/gustofied/kalkulator>
- Live: <https://gustofied.github.io/kalkulator/>
- App: static Vite/TypeScript, deployed to GitHub Pages by `.github/workflows/pages.yml`
- Execution: one model-specific WebGPU runtime in a dedicated Web Worker
- Artifact: 19 immutable full-BF16 binary shards, about 1.93 GiB total
- Cache: each complete shard is stored in OPFS; a warm load reads the local copy instead of downloading it again
- Interface: editorial black/violet UI with GSAP motion and KaTeX rendering

There is no production JAX.js path, runtime probe, A/B switch, or fallback engine. [`src/model.ts`](../src/model.ts) remains the JAX.js reference implementation and architecture oracle.

## Pinned model contract

The converter and loader accept one upstream checkpoint:

- Repository: `paradigma-inc/limite-1b-violetto`
- Revision: `b1f3d572ccacb6919f4d64c321b70ba034ddaef2`
- File: `model.safetensors`
- Source SHA-256: `9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5`

The production manifest is pinned to the immutable Hugging Face commit `df708b2170cecfde605a77dd57cd5bb78f36c852` under `webgpu-full-bf16-v16`. The loader validates the source identity, model constants, tensor inventory, shard sizes, and tensor ranges before using the artifact.

The model contract is 48 layers with hidden size 1,280, intermediate size 3,328, 10 query heads, 2 key/value heads, and head dimension 128. Local attention retains the current token plus 1,024 preceding tokens; layers 3, 7, …, 47 use global attention. RoPE covers the first 64 dimensions of each head with base 1,024. Value embeddings occur on layers 1, 4, …, 46, XSA is used on every layer, MUDD mixing occurs at layers 24 and 47, and the token embedding is tied to the vocabulary head. Logits use Paradigma's soft cap `23 * sigmoid((raw + 5) / 7.5)`.

The browser artifact keeps every model projection and tied embedding in BF16, with FP32 control tensors. Conversion fuses Q/K/V and gate/up matrices, folds the published projection scales, stores the XSA coefficients after `tanh`, and includes exact BF16 lookup tables for GELU, SiLU, sigmoid, and sampling. This keeps the browser path aligned with the reference computation while avoiding runtime table generation.

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

Generation uses temperature 0.6 and top-p 0.95 on the GPU. Token ID 151643 is BOS, EOS, and PAD; `<|im_end|>` is not treated as EOS.

The runtime has a 32,768-token working context. The maximum output is the unused portion after the formatted prompt. Generation ends on EOS, when the context is full, or after a complete balanced `\boxed{...}` answer. If the model emits `<think>`, the box is accepted only after `</think>`; when the answer is inside `\[...\]` or `\(...\)`, the closing math delimiter is retained before stopping.

## Loading and performance

The first visit downloads about 1.93 GiB of model data. Successful shards are cached in OPFS, stale artifact shards are removed, and later visits load the pinned artifact locally. The browser still has to read the cached bytes, create GPU buffers, and compile pipelines on each page load. If the browser cannot grant enough storage, inference still works for that visit but the weights may need to be downloaded again later.

A local WebGPU smoke test of the full-BF16 path returned the correct boxed answer at 31.31 generated tokens per second. This is not a portable benchmark; performance depends on the GPU, browser, thermals, prompt length, and generated sequence length.

## Deployment invariants

- Keep the app browser-only and compatible with static GitHub Pages hosting.
- Do not commit model weights to the application repository.
- Keep all production artifact URLs revision-pinned.
- Preserve `src/model.ts` as the readable JAX.js reference when changing the custom runtime.
- Verify both output correctness and inference in a real WebGPU browser after runtime changes.
