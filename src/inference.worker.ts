import {
  LIMITE_CONTEXT_TOKENS,
  LIMITE_TOKENIZER_URL,
  LimiteWebGpuEngine,
} from "./limite-engine/engine";
import { hasCompleteFinalBox } from "./boxed-answer";
import type {
  CompletionReason,
  InferenceRequest,
  RunTimings,
} from "./inference-protocol";
import { ViolettoTokenizer } from "./tokenizer";

const RENDER_INTERVAL_MS = 250;

type GenerationResult = {
  readonly text: string;
  readonly tokens: number;
  readonly measuredTokens: number;
  readonly generationMs: number;
  readonly reason: CompletionReason;
};

let engine: LimiteWebGpuEngine | null = null;
let tokenizer: ViolettoTokenizer | null = null;
let busy = false;
let activeRunId: number | null = null;
let lastAcceptedRunId = 0;
const cancelledRunIds = new Set<number>();

self.addEventListener("message", (event: MessageEvent<InferenceRequest>) => {
  const request = event.data;
  if (request.type === "cancel") {
    if (request.runId === activeRunId) cancelledRunIds.add(request.runId);
    return;
  }

  if (request.type === "solve") {
    if (!Number.isSafeInteger(request.runId) || request.runId <= 0) {
      rejectRun(request.runId, "Invalid inference run id.");
      return;
    }
    if (busy) {
      rejectRun(request.runId, "Violetto is already solving another problem.");
      return;
    }
    if (request.runId <= lastAcceptedRunId) {
      rejectRun(request.runId, "That inference run is stale.");
      return;
    }

    lastAcceptedRunId = request.runId;
    activeRunId = request.runId;
  } else if (busy) {
    return;
  }

  busy = true;
  const operation =
    request.type === "prepare"
      ? prepare()
      : solve(request.runId, request.problem);
  void operation.finally(() => {
    if (request.type === "solve") {
      cancelledRunIds.delete(request.runId);
      if (activeRunId === request.runId) activeRunId = null;
    }
    busy = false;
  });
});

async function prepare(): Promise<void> {
  const timings: RunTimings = {};
  try {
    await setup(timings);
    postMessage({ type: "ready", timings });
  } catch (error) {
    reportError(error);
  }
}

async function solve(
  runId: number,
  problem: string,
): Promise<void> {
  const timings: RunTimings = {};
  const runStarted = performance.now();

  try {
    await setup(timings, runId);
    const activeEngine = engine!;
    const activeTokenizer = tokenizer!;
    const tokens = activeTokenizer.encode(formatPrompt(problem));
    if (tokens.length >= LIMITE_CONTEXT_TOKENS) {
      throw new Error("That problem is too long for the browser context.");
    }

    const outputBudget = LIMITE_CONTEXT_TOKENS - tokens.length;

    postStatus(`prefill · ${tokens.length} tok`, undefined, runId);
    const prefillStarted = performance.now();
    const first = await activeEngine.prefill(tokens);
    const firstTokenAt = performance.now();
    timings.prefillMs = firstTokenAt - prefillStarted;
    timings.firstTokenMs = firstTokenAt - runStarted;

    const generation = await generate(
      runId,
      activeEngine,
      activeTokenizer,
      first,
      outputBudget,
      firstTokenAt,
    );
    timings.totalMs = performance.now() - runStarted;
    const speed =
      generation.generationMs > 0
        ? generation.measuredTokens / (generation.generationMs / 1000)
        : 0;
    postMessage({
      type: "done",
      runId,
      text: generation.text,
      tokens: generation.tokens,
      speed,
      reason: generation.reason,
      timings,
    });
  } catch (error) {
    reportError(error, runId);
  }
}

async function generate(
  runId: number,
  activeEngine: LimiteWebGpuEngine,
  activeTokenizer: ViolettoTokenizer,
  first: number,
  outputBudget: number,
  firstTokenAt: number,
): Promise<GenerationResult> {
  const sampled: number[] = [];
  const decoder = activeTokenizer.createDecoder();
  let decoded = "";
  let postedLength = 0;
  let lastRender = 0;
  let reason: CompletionReason | null = null;
  let pending: readonly number[] = [first];
  const tokenLimit = Math.min(
    Math.max(Math.floor(outputBudget), 1),
    Math.max(LIMITE_CONTEXT_TOKENS - activeEngine.position, 1),
  );

  const appendToken = (token: number): void => {
    decoded = decoder.push(token);
    const now = performance.now();
    const measuredTokens = Math.max(0, sampled.length - 1);
    const elapsed = Math.max((now - firstTokenAt) / 1000, 0.001);
    const speed = measuredTokens / elapsed;

    if (
      now - lastRender >= RENDER_INTERVAL_MS ||
      sampled.length === 1 ||
      sampled.length === tokenLimit
    ) {
      const delta = decoded.slice(postedLength);
      if (delta) {
        postMessage({
          type: "update",
          runId,
          delta,
          tokens: sampled.length,
          speed,
        });
        postedLength = decoded.length;
      }
      lastRender = now;
    }
  };

  generation: while (sampled.length < tokenLimit) {
    if (cancelledRunIds.has(runId)) {
      reason = "cancelled";
      break;
    }
    for (const next of pending) {
      if (cancelledRunIds.has(runId)) {
        reason = "cancelled";
        break generation;
      }
      if (next === activeTokenizer.eosToken) {
        reason = "eos";
        break generation;
      }

      sampled.push(next);
      appendToken(next);

      if (hasCompleteFinalBox(decoded)) {
        reason = "boxed";
        break generation;
      }
      if (sampled.length >= tokenLimit) break generation;
    }

    const remaining = tokenLimit - sampled.length;
    if (remaining <= 0) break;
    pending = await activeEngine.decodeBatch(
      sampled[sampled.length - 1],
      Math.min(4, remaining),
    );
  }

  const generationMs = performance.now() - firstTokenAt;
  return {
    text: decoder.finish(),
    tokens: sampled.length,
    measuredTokens: Math.max(0, sampled.length - 1),
    generationMs,
    reason: reason ?? "limit",
  };
}

async function setup(timings: RunTimings, runId?: number): Promise<void> {
  if (engine && tokenizer) return;

  const tokenizerTask = tokenizer
    ? Promise.resolve(tokenizer)
    : loadTokenizer(timings, runId);
  const engineTask = engine
    ? Promise.resolve(engine)
    : LimiteWebGpuEngine.create((progress) => {
        if (progress.phase === "manifest") {
          postStatus("preparing weights", undefined, runId);
          return;
        }
        const ratio = progress.loadedBytes / progress.totalBytes;
        const action =
          progress.source === "cache" ? "reading cached weights" : "downloading weights";
        postStatus(
          `${action} · ${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.totalBytes)}`,
          ratio,
          runId,
        );
      });

  const [loadedTokenizer, loadedEngine] = await Promise.all([tokenizerTask, engineTask]);
  tokenizer = loadedTokenizer;
  engine = loadedEngine;
  timings.deviceMs = loadedEngine.timings.deviceMs;
  timings.artifactMs = loadedEngine.timings.artifactMs;
  timings.pipelineMs = loadedEngine.timings.pipelineMs;
}

async function loadTokenizer(
  timings: RunTimings,
  runId?: number,
): Promise<ViolettoTokenizer> {
  postStatus("preparing language", undefined, runId);
  const started = performance.now();
  const response = await fetch(LIMITE_TOKENIZER_URL, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Tokenizer download failed (${response.status}).`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  timings.tokenizerReadMs = performance.now() - started;
  const parseStarted = performance.now();
  const loaded = ViolettoTokenizer.fromBinary(bytes);
  timings.tokenizerParseMs = performance.now() - parseStarted;
  return loaded;
}

function formatPrompt(problem: string): string {
  const instruction =
    "Please reason step by step, and put your final answer within \\boxed{}.";
  return `<|endoftext|><|im_start|>system\nYou are a helpful assistant.\n${instruction}<|im_end|>\n<|im_start|>user\n${problem}<|im_end|>\n<|im_start|>assistant\n`;
}

function reportError(error: unknown, runId?: number): void {
  console.error(error);
  postMessage({
    type: "error",
    message: error instanceof Error ? error.message : String(error),
    runId,
  });
}

function rejectRun(runId: number, message: string): void {
  postMessage({ type: "rejected", runId, message });
}

function postStatus(status: string, progress?: number, runId?: number): void {
  postMessage({ type: "status", status, progress, runId });
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MiB`;
  return `${Math.round(bytes / 1024)} KiB`;
}

export {};
