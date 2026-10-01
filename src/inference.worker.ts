import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  tree,
} from "@jax-js/jax";
import { cachedFetch, opfs } from "@jax-js/loaders";

import type { WebGpuSampler } from "./gpu-sampler";
import {
  createState,
  loadModel,
  lmHeadWeight,
  logitsFromHidden,
  MODEL_BYTES,
  MODEL_URL,
  prefill,
  step,
  TOKENIZER_URL,
  type LimiteModel,
} from "./model";
import { ViolettoTokenizer } from "./tokenizer";

const INITIAL_CACHE_TOKENS = 384;
const HARD_TOKEN_LIMIT = 768;
const TEMPERATURE = 0.6;
const TOP_K = 50;
const TOP_P = 0.95;
const RENDER_INTERVAL_MS = 250;

type InboundMessage =
  | { type: "prepare" }
  | {
      type: "solve";
      problem: string;
      options?: { lmHead?: "jax" | "wgsl"; seed?: number };
    };

type RunTimings = {
  deviceMs?: number;
  tokenizerReadMs?: number;
  tokenizerParseMs?: number;
  modelReadMs?: number;
  modelUploadMs?: number;
  prefillMs?: number;
  firstTokenMs?: number;
  totalMs?: number;
};

let model: LimiteModel | null = null;
let tokenizer: ViolettoTokenizer | null = null;
let initialized = false;
let busy = false;
let gpuSampler: WebGpuSampler | null = null;
let gpuSamplerPromise: Promise<WebGpuSampler | null> | null = null;

self.addEventListener("message", (event: MessageEvent<InboundMessage>) => {
  if (busy) return;
  busy = true;
  if (event.data.type === "prepare") {
    void prepare().finally(() => {
      busy = false;
    });
    return;
  }
  void solve(event.data.problem, event.data.options).finally(() => {
    busy = false;
  });
});

async function prepare(): Promise<void> {
  const timings: RunTimings = {};
  try {
    await setup(timings);
    postMessage({ type: "ready", timings });
  } catch (error) {
    console.error(error);
    postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

async function solve(
  problem: string,
  options?: { lmHead?: "jax" | "wgsl"; seed?: number },
): Promise<void> {
  const timings: RunTimings = {};
  const runStarted = performance.now();

  try {
    await setup(timings);
    const activeModel = model!;
    const activeTokenizer = tokenizer!;
    const tokens = activeTokenizer.encode(formatPrompt(problem));
    const state = createState(tokens.length + INITIAL_CACHE_TOKENS);
    const generated: number[] = [];
    const decoder = activeTokenizer.createDecoder();
    const speeds: number[] = [];
    let decoded = "";
    let lastSpeed = 0;
    let hidden: np.Array | null = null;
    let lastRender = 0;
    let reason: "complete" | "limit" = "limit";
    const random = makeRandom(options?.seed);

    try {
      postStatus(`prefill · ${tokens.length} tok`);
      const prefillStarted = performance.now();
      hidden = prefill(
        tree.ref(activeModel),
        np.array(tokens, { dtype: np.uint32 }),
        state,
      );
      await blockUntilReady(hidden);
      timings.prefillMs = performance.now() - prefillStarted;
      const decodeStarted = performance.now();

      for (let index = 0; index < HARD_TOKEN_LIMIT; index++) {
        const current = hidden;
        if (!current) throw new Error("Missing decode hidden state.");
        hidden = null;
        const next = await sampleHidden(
          activeModel,
          current,
          TEMPERATURE,
          TOP_K,
          TOP_P,
          random(),
          options?.lmHead ?? "jax",
        );
        if (next === activeTokenizer.eosToken || next === activeTokenizer.imEndToken) {
          reason = "complete";
          break;
        }

        generated.push(next);
        decoded = decoder.push(next);
        const now = performance.now();
        timings.firstTokenMs ??= now - runStarted;
        const elapsed = (now - decodeStarted) / 1000;
        const speed = generated.length / Math.max(elapsed, 0.001);
        lastSpeed = speed;

        if (
          now - lastRender >= RENDER_INTERVAL_MS ||
          generated.length === 1 ||
          index === HARD_TOKEN_LIMIT - 1
        ) {
          speeds.push(speed);
          postMessage({
            type: "update",
            text: decoded,
            tokens: generated.length,
            speed,
            speeds: [...speeds],
          });
          lastRender = now;
        }

        if (hasCompleteFinalAnswer(decoded)) {
          reason = "complete";
          break;
        }

        if (index + 1 < HARD_TOKEN_LIMIT) {
          hidden = step(tree.ref(activeModel), next, state);
        }
      }

      const text = decoder.finish();
      timings.totalMs = performance.now() - runStarted;
      postMessage({
        type: "done",
        text,
        tokens: generated.length,
        speed: lastSpeed,
        speeds,
        reason,
        timings,
      });
    } finally {
      hidden?.dispose();
      tree.dispose(state);
    }
  } catch (error) {
    console.error(error);
    postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}

async function setup(timings: RunTimings): Promise<void> {
  if (!initialized) {
    postStatus("starting WebGPU");
    const started = performance.now();
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      throw new Error("WebGPU is unavailable in this browser.");
    }
    defaultDevice("webgpu");
    initialized = true;
    timings.deviceMs = performance.now() - started;
  }

  if (!tokenizer) {
    const started = performance.now();
    const bytes = await fetchAsset(TOKENIZER_URL, "tokenizer");
    timings.tokenizerReadMs = performance.now() - started;
    postStatus("preparing tokenizer");
    const parseStarted = performance.now();
    tokenizer = ViolettoTokenizer.fromBinary(bytes);
    timings.tokenizerParseMs = performance.now() - parseStarted;
  }

  if (!model) {
    const started = performance.now();
    const checkpoint = await fetchAsset(MODEL_URL, "model", MODEL_BYTES);
    timings.modelReadMs = performance.now() - started;
    postStatus("uploading weights");
    const uploadStarted = performance.now();
    model = await loadModel(checkpoint);
    timings.modelUploadMs = performance.now() - uploadStarted;
  }
}

async function fetchAsset(
  url: string,
  label: string,
  expectedBytes?: number,
): Promise<Uint8Array<ArrayBuffer>> {
  try {
    let cached = await opfs.info(url);
    if (cached && expectedBytes && cached.size !== expectedBytes) {
      postStatus(`repairing ${label} cache`);
      await opfs.remove(url);
      cached = null;
    }
    postStatus(cached ? `reading cached ${label}` : `downloading ${label}`);
    let lastProgress = 0;
    return await cachedFetch(url, undefined, ({ loadedBytes, totalBytes }) => {
      if (cached || performance.now() - lastProgress < 120) return;
      lastProgress = performance.now();
      const progress = totalBytes ? loadedBytes / totalBytes : undefined;
      const detail = totalBytes
        ? `${formatBytes(loadedBytes)} / ${formatBytes(totalBytes)}`
        : formatBytes(loadedBytes);
      postStatus(`downloading ${label} · ${detail}`, progress);
    });
  } catch (error) {
    console.warn("OPFS cache unavailable; using a direct fetch", error);
    postStatus(`downloading ${label}`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  }
}

function formatPrompt(problem: string): string {
  const system =
    "You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}.";
  return `<|endoftext|><|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${problem}<|im_end|>\n<|im_start|>assistant\n`;
}

function hasCompleteFinalAnswer(text: string): boolean {
  const thoughtStart = text.lastIndexOf("<think>");
  const thoughtEnd = text.lastIndexOf("</think>");
  if (thoughtStart >= 0 && thoughtEnd < thoughtStart) return false;
  const visible = thoughtEnd >= 0 ? text.slice(thoughtEnd + 8) : text;
  const box = visible.lastIndexOf("\\boxed{");
  if (box < 0) return false;

  let depth = 0;
  for (let index = box + 6; index < visible.length; index++) {
    if (visible[index] === "{") depth++;
    if (visible[index] === "}" && --depth === 0) {
      return hasBalancedMath(visible);
    }
  }
  return false;
}

function hasBalancedMath(text: string): boolean {
  if (count(text, "\\[") !== count(text, "\\]")) return false;
  if (count(text, "\\(") !== count(text, "\\)")) return false;
  const withoutDisplays = text.replaceAll("$$", "");
  return count(withoutDisplays, "$") % 2 === 0 && count(text, "$$") % 2 === 0;
}

function count(text: string, needle: string): number {
  return text.split(needle).length - 1;
}

async function sampleHidden(
  activeModel: LimiteModel,
  hidden: np.Array,
  temperature: number,
  topK: number,
  topP: number,
  randomValue: number,
  strategy: "jax" | "wgsl",
): Promise<number> {
  if (strategy === "wgsl") {
    const sampler = await getGpuSampler();
    if (sampler) {
      return sampleCustomHead(
        sampler,
        hidden,
        lmHeadWeight(activeModel),
        temperature,
        topP,
        randomValue,
      );
    }
  }

  return sampleCpu(
    logitsFromHidden(activeModel, hidden),
    temperature,
    topK,
    topP,
    randomValue,
  );
}

async function getGpuSampler(): Promise<WebGpuSampler | null> {
  if (gpuSampler) return gpuSampler;
  if (gpuSamplerPromise) return gpuSamplerPromise;

  gpuSamplerPromise = (async () => {
    try {
      const { WebGpuSampler: Sampler } = await import("./gpu-sampler");
      const candidate = await Sampler.create();
      await candidate.selfTest();
      if (!candidate.supportsLmHead) {
        console.warn("Experimental WGSL head is unsupported; using JAX.");
        return null;
      }
      gpuSampler = candidate;
      return candidate;
    } catch (error) {
      console.warn("Experimental WGSL head unavailable; using JAX.", error);
      return null;
    }
  })();
  return gpuSamplerPromise;
}

async function sampleCustomHead(
  sampler: WebGpuSampler,
  hidden: np.Array,
  weight: np.Array,
  temperature: number,
  topP: number,
  randomValue: number,
): Promise<number> {
  const keepAlive = hidden.ref;
  try {
    await blockUntilReady(hidden);
    return await sampler.sampleLmHead(
      hidden,
      weight.ref,
      temperature,
      topP,
      randomValue,
    );
  } finally {
    keepAlive.dispose();
  }
}

function makeRandom(seed?: number): () => number {
  if (seed === undefined || !Number.isFinite(seed)) return Math.random;
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

async function sampleCpu(
  logits: np.Array,
  temperature: number,
  topK: number,
  topP: number,
  randomValue: number = Math.random(),
): Promise<number> {
  const values = (await logits.data()) as Float32Array;
  return sampleValues(values, temperature, topK, topP, randomValue);
}

function sampleValues(
  values: Float32Array,
  temperature: number,
  topK: number,
  topP: number,
  randomValue: number,
): number {
  const candidates: { id: number; logit: number }[] = [];

  for (let id = 0; id < values.length; id++) {
    const logit = values[id];
    if (!Number.isFinite(logit)) continue;
    if (candidates.length < topK || logit > candidates[candidates.length - 1].logit) {
      let at = 0;
      while (at < candidates.length && logit < candidates[at].logit) at++;
      candidates.splice(at, 0, { id, logit });
      if (candidates.length > topK) candidates.pop();
    }
  }
  if (!candidates.length) throw new Error("The model returned no finite logits.");
  if (temperature <= 0) return candidates[0].id;

  const maximum = softcap(candidates[0].logit);
  const probabilities = candidates.map(({ logit }) =>
    Math.exp((softcap(logit) - maximum) / temperature),
  );
  const total = probabilities.reduce((sum, probability) => sum + probability, 0);
  let keptTotal = 0;
  let kept = 0;
  while (kept < candidates.length) {
    keptTotal += probabilities[kept++];
    if (keptTotal / total >= topP) break;
  }
  let pick = randomValue * keptTotal;
  for (let index = 0; index < kept; index++) {
    pick -= probabilities[index];
    if (pick <= 0) return candidates[index].id;
  }
  return candidates[kept - 1].id;
}

function softcap(logit: number): number {
  return 23 / (1 + Math.exp(-((logit + 5) / 7.5)));
}

function postStatus(status: string, progress?: number): void {
  postMessage({ type: "status", status, progress });
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}

export {};
