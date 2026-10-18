import { blockUntilReady, defaultDevice, getWebGPUDevice, init, numpy as np, random, tree } from "@jax-js/jax";
import { LIMITE_CONTEXT_TOKENS, LIMITE_EOS_TOKEN, LIMITE_TOKENIZER_VOCAB_SIZE } from "./limite-config";
import { createState, disposeState, loadQ4Model, logitsFromHidden, prefill, step, type LimiteModel } from "./model";
import type { LimiteArtifactProgressCallback } from "./model-artifact";
import { sampleToken } from "./jax-sampler";

/** JAX.js owns tensor execution and generates all GPU kernels. */
export class LimiteJaxEngine {
  readonly device = getWebGPUDevice();
  readonly timings;
  readonly #model: LimiteModel;
  #state = createState();
  #key = random.key(crypto.getRandomValues(new Uint32Array(1))[0]);

  private constructor(model: LimiteModel, timings: { deviceMs: number; artifactMs: number }) {
    this.#model = model;
    this.timings = timings;
  }

  static async create(onProgress?: LimiteArtifactProgressCallback): Promise<LimiteJaxEngine> {
    const started = performance.now();
    const available = await init("webgpu");
    if (!available.includes("webgpu")) throw new Error("WebGPU is unavailable in this browser.");
    defaultDevice("webgpu");
    if (!getWebGPUDevice().features.has("shader-f16")) {
      throw new Error("This browser needs WebGPU with shader-f16 support.");
    }
    const deviceMs = performance.now() - started;
    const artifactStarted = performance.now();
    const model = await loadQ4Model(onProgress);
    return new LimiteJaxEngine(model, { deviceMs, artifactMs: performance.now() - artifactStarted });
  }

  get position(): number { return this.#state.position; }

  #sample(hidden: np.Array): np.Array {
    const [token, nextKey] = sampleToken(logitsFromHidden(this.#model, hidden), this.#key);
    this.#key = nextKey;
    return token;
  }

  async prefill(tokens: readonly number[], onProgress: (processed: number) => void = () => {}): Promise<number> {
    if (!tokens.length || tokens.length >= LIMITE_CONTEXT_TOKENS) throw new Error("The problem exceeds the browser context.");
    disposeState(this.#state);
    this.#state = createState();
    this.#key.dispose();
    this.#key = random.key(crypto.getRandomValues(new Uint32Array(1))[0]);
    onProgress(0);
    // Bound prefill's quadratic attention workspace. Longer prompts continue
    // through the same cache, without allocating a full-context score matrix.
    const initial = Math.min(tokens.length, 256);
    let hidden = prefill(this.#model, np.array(tokens.slice(0, initial), { dtype: np.uint32 }), this.#state);
    for (let index = initial; index < tokens.length; index++) {
      await blockUntilReady([hidden, this.#state.caches]);
      hidden.dispose();
      if (index % 32 === 0) onProgress(index);
      hidden = step(this.#model, tokens[index], this.#state);
    }
    const result = readTokens(await this.#sample(hidden).data())[0];
    onProgress(tokens.length);
    return result;
  }

  async decodeBatch(token: number, count = 1): Promise<readonly number[]> {
    const size = Math.min(count, LIMITE_CONTEXT_TOKENS - this.position);
    if (!Number.isInteger(size) || size < 1 || size > 4) throw new Error("Invalid decode batch size.");
    const tokens: np.Array[] = [];
    let next: np.Array | null = np.array(token, { dtype: np.uint32 });
    try {
      for (let index = 0; index < size; index++) {
        const input: np.Array = next!;
        next = null; // step consumes the input token.
        next = this.#sample(step(this.#model, input, this.#state));
        tokens.push(next.ref);
        // Submit through JAX's public API without a CPU readback or per-kernel
        // preparation promises. This runs in the worker; the batch read below
        // still yields while the GPU finishes. Keep graphs and scratch bounded.
        next.ref.gpuBufferSync();
      }
      const batch = np.stack(tokens);
      tokens.length = 0; // stack takes ownership of its inputs.
      return readTokens(await batch.data());
    } finally {
      next?.dispose();
      tree.dispose(tokens);
    }
  }

  destroy(): void {
    disposeState(this.#state);
    this.#key.dispose();
    tree.dispose(this.#model);
  }
}

function readTokens(values: ArrayLike<number>): number[] {
  const tokens: number[] = [];
  for (let index = 0; index < values.length; index++) {
    const token = Number(values[index]);
    if (!Number.isInteger(token) || token < 0 || token >= LIMITE_TOKENIZER_VOCAB_SIZE) {
      throw new Error("The model produced invalid numerical output. Reload to continue.");
    }
    tokens.push(token);
    // Later slots may already have executed, but nothing after EOS is exposed.
    // A new problem resets the cache and PRNG; a completed answer cannot resume.
    if (token === LIMITE_EOS_TOKEN) break;
  }
  return tokens;
}
