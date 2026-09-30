# Kalkulator project brief

## Goal

Make Violetto Limite 1B usable as a private, browser-only mathematics playground: no inference server, no submitted prompts, and no account requirement.

## Current state

- Source: <https://github.com/gustofied/kalkulator>
- Live: <https://gustofied.github.io/kalkulator/>
- Runtime: JAX.js on WebGPU
- Interface: static Vite/TypeScript, KaTeX for mathematics, D3 for generation speed
- Model: 1.93 GiB FP16 safetensors hosted separately on Hugging Face
- Deployment: GitHub Pages through `.github/workflows/pages.yml`

The deployed reference implementation has completed end-to-end browser inference and solved `x + 3 = 8` correctly. It has no backend.

## How it got here

The upstream Violetto checkpoint is BF16. The browser artifact was converted to FP16 and its custom architecture was ported directly to JAX.js. The port includes 48 transformer layers, global and sliding-window attention, partial interleaved RoPE, value embeddings and gates, XSA, residual lambdas, and the two MUDD mixing points. The embedding matrix is tied to the language-model head.

JAX.js was chosen because it made an exact custom architecture possible quickly. It is the reference implementation, not yet a commitment to the final optimized runtime.

## Observed behavior

- Warm inference on an M2 MacBook Air with 16 GB unified memory reached roughly 6–7 tokens/second around 200 generated tokens.
- A long 700+ token generation degraded to roughly 2 tokens/second.
- The UI currently says `downloading` before checking Cache Storage, so a warm cache read looks like a network download.
- On every page load, cached weights still need to be read, parsed, uploaded to the GPU, and compiled.
- Vague questions can trigger an unnecessarily long worked example; stopping and response control need work.

## Known bottlenecks

1. The 1.93 GiB FP16 model is not quantized.
2. Each token reads the full vocabulary logits back from the GPU and samples in JavaScript.
3. Decode crosses the JavaScript/GPU boundary layer by layer.
4. KV-cache updates touch capacity-sized tensors instead of updating one slot efficiently.
5. The full answer, KaTeX, and D3 trace are redrawn for every generated token.
6. Generation allows up to 1,024 tokens even for simple prompts.
7. Cache status and load-stage timings are not instrumented.

## Optimization sequence

1. Replace the custom Cache API wrapper with `@jax-js/loaders` OPFS caching and show truthful cache/download/upload/compile states.
2. Add local-only timings for cache read, download, parse, GPU upload, compile, prefill, and decode.
3. Tighten prompting, stopping, and default output limits.
4. Throttle UI rendering and move inference into a Web Worker.
5. Keep top-k/top-p sampling on the GPU and transfer only the selected data.
6. Replace capacity-wide KV-cache rewrites with indexed updates.
7. Reduce GPU submissions by fusing more of the decode pass.
8. Evaluate INT8/INT4 weights.
9. Build one ONNX Runtime/Transformers.js comparison before deciding whether to leave JAX.js.

## Runtime decision

Keep JAX.js until measurements justify a migration. Transformers.js/ONNX Runtime is the first comparison candidate because it offers WebGPU and quantized model formats. WebLLM/MLC is optimized for browser LLMs but supporting Violetto's custom architecture would be a larger port.

## Definition of progress

Every optimization should report cold-load bytes and time, warm-load time, time to first token, steady decode rate, long-context decode rate, peak memory observations, output correctness, and browser/hardware details.

