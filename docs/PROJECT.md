# Kalkulator project brief

## Goal

Make Violetto Limite 1B usable as a private, browser-only mathematics playground: no inference server, no submitted prompts, and no account requirement.

## Current state

- Source: <https://github.com/gustofied/kalkulator>
- Live: <https://gustofied.github.io/kalkulator/>
- Runtime: JAX.js on WebGPU
- Interface: static Vite/TypeScript with GSAP motion and KaTeX mathematics
- Model: 1.93 GiB FP16 safetensors hosted separately on Hugging Face
- Deployment: GitHub Pages through `.github/workflows/pages.yml`
- Cache: `@jax-js/loaders` OPFS storage with validated checkpoint size
- Execution: WebGPU inference in a dedicated Web Worker
- Startup: the worker prepares the device, tokenizer, sampler, and model before revealing the prompt
- Sampling: specialized WGSL hierarchical top-k plus temperature/top-p selection, with CPU verification and fallback
- Decode layout: fused QKV/gate-up/gate projections, shared RoPE factors, and packed KV tensors
- Streaming: incremental UTF-8 token decoding and 250 ms UI render batches

The deployed reference implementation has completed end-to-end browser inference and solved `x + 3 = 8` correctly. It has no backend.

## How it got here

The upstream Violetto checkpoint is BF16. The browser artifact was converted to FP16 and its custom architecture was ported directly to JAX.js. The port includes 48 transformer layers, global and sliding-window attention, partial interleaved RoPE, value embeddings and gates, XSA, residual lambdas, and the two MUDD mixing points. The embedding matrix is tied to the language-model head.

JAX.js was chosen because it made an exact custom architecture possible quickly. It is the reference implementation, not yet a commitment to the final optimized runtime.

## Observed behavior

- Warm inference on an M2 MacBook Air with 16 GB unified memory reached roughly 6–7 tokens/second around 200 generated tokens.
- A long 700+ token generation degraded to roughly 2 tokens/second.
- A September 2026 warm-cache verification in Chrome produced the correct answer to `x + 3 = 8` in 343 generated tokens: 1.1 seconds prefill, 4.2 seconds to first token, and 7.2 tokens/second average. This is one local run, not a controlled benchmark.
- The specialized sampler's first-run probe took 12.9 ms on WebGPU versus 16.8 ms for the JavaScript reference, returned the same token, and sustained 8.1 tokens/second during a 149-token warm run. These are local spot checks, not a controlled benchmark.
- After projection fusion, shared RoPE, packed KV updates, and sampler-side softcap, a warm Chrome run on the same M2 Air produced the correct boxed answer in 403 generated tokens at 10.6 tokens/second with 1.4 seconds of prefill. The earlier comparable UI run was 7.8 tokens/second, so this spot check is about 36% faster.
- A patched wllama/llama.cpp WebGPU prototype with the published Q4_K_M GGUF reached 12.5–13.8 tokens/second on the same machine. The Q2_K_L build reached 11.5 tokens/second and multithreaded Wasm reached roughly 2 tokens/second. Q4 needed 321–384 tokens for the toy equation and one run hit its output cap before the boxed answer, so the raw-rate improvement did not justify replacing the reference runtime. These are local spot checks, not a controlled benchmark.
- A timestamp-query profile showed the fused gate/up matvec and vocabulary head as the largest GPU kernels. It also showed that decode still launches hundreds of small kernels per token, leaving submission overhead as the main target after quantization.
- A completed OPFS checkpoint survives reload without another network download. Warm loads report the local read separately from GPU upload.
- On every page load, cached weights still need to be read, parsed, uploaded to the GPU, and compiled.
- Violetto's exact canonical system prompt from `chat_template.jinja` is required; custom system wording caused runaway hidden reasoning in testing.
- Generation runs in a worker, streams draft work into a collapsible surface, renders the final answer separately, supports Stop, stops after a balanced boxed answer, and has a 768-token hard safety limit.

## Known bottlenecks

1. The 1.93 GiB FP16 model is not quantized.
2. Decode crosses the JavaScript/GPU boundary layer by layer.
3. Packed KV-cache updates still touch capacity-sized tensors instead of updating one slot efficiently.
4. Sampling still requires a four-byte GPU readback and queue synchronization for every token.
5. The throttled answer, KaTeX, and D3 trace still rebuild their rendered output during generation.
6. A small prompt can still spend hundreds of tokens in hidden reasoning before producing its answer.
7. Load and inference timings are visible locally but are not yet collected into a repeatable benchmark harness.

## Optimization sequence

1. Build a repeatable cold/warm benchmark harness from the current local timings.
2. Replace capacity-wide KV-cache rewrites with indexed updates when JAX.js exposes an efficient update primitive.
3. Reduce GPU submissions by fusing more of the decode pass; queue-level batching alone is unsafe because JAX.js releases intermediate buffers after each submission.
4. Evaluate INT8/INT4 weights and a specialized quantized matmul kernel.
5. Build one ONNX Runtime/Transformers.js comparison before deciding whether to leave JAX.js.

## Runtime decision

Keep JAX.js until measurements justify a migration. Generic Q4 execution in a patched wllama/llama.cpp build was only modestly faster and less reliable for time-to-answer; a worthwhile quantized path needs a specialized fused decode graph, not just smaller weights. Transformers.js/ONNX Runtime remains a comparison candidate because it offers WebGPU and quantized model formats. WebLLM/MLC is optimized for browser LLMs but supporting Violetto's custom architecture would be a larger port.

## Definition of progress

Every optimization should report cold-load bytes and time, warm-load time, time to first token, steady decode rate, long-context decode rate, peak memory observations, output correctness, and browser/hardware details.
