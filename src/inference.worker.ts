import {
  LIMITE_CONTEXT_TOKENS,
  LIMITE_TOKENIZER_URL,
  LimiteWebGpuEngine,
} from "./limite-engine/engine";
import { ViolettoTokenizer } from "./tokenizer";

const RENDER_INTERVAL_MS = 250;

type InboundMessage =
  | { type: "prepare" }
  | { type: "solve"; problem: string };

type RunTimings = {
  deviceMs?: number;
  tokenizerReadMs?: number;
  tokenizerParseMs?: number;
  artifactMs?: number;
  pipelineMs?: number;
  prefillMs?: number;
  firstTokenMs?: number;
  totalMs?: number;
};

let engine: LimiteWebGpuEngine | null = null;
let tokenizer: ViolettoTokenizer | null = null;
let busy = false;

self.addEventListener("message", (event: MessageEvent<InboundMessage>) => {
  if (busy) return;
  busy = true;
  const operation = event.data.type === "prepare" ? prepare() : solve(event.data.problem);
  void operation.finally(() => {
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

async function solve(problem: string): Promise<void> {
  const timings: RunTimings = {};
  const runStarted = performance.now();

  try {
    await setup(timings);
    const activeEngine = engine!;
    const activeTokenizer = tokenizer!;
    const tokens = activeTokenizer.encode(formatPrompt(problem));
    if (tokens.length >= LIMITE_CONTEXT_TOKENS) {
      throw new Error("That problem is too long for the browser context.");
    }

    const generated: number[] = [];
    const decoder = activeTokenizer.createDecoder();
    let decoded = "";
    let lastSpeed = 0;
    let lastRender = 0;
    let reason: "complete" | "limit" = "limit";

    postStatus(`prefill · ${tokens.length} tok`);
    const prefillStarted = performance.now();
    const first = await activeEngine.prefill(tokens);
    const firstTokenAt = performance.now();
    timings.prefillMs = firstTokenAt - prefillStarted;
    timings.firstTokenMs = firstTokenAt - runStarted;

    const outputBudget = LIMITE_CONTEXT_TOKENS - tokens.length;
    let pending: readonly number[] = [first];
    generation: while (generated.length < outputBudget) {
      for (const next of pending) {
        if (next === activeTokenizer.eosToken) {
          reason = "complete";
          break generation;
        }

        generated.push(next);
        decoded = decoder.push(next);
        const now = performance.now();
        const measuredTokens = Math.max(0, generated.length - 1);
        const elapsed = Math.max((now - firstTokenAt) / 1000, 0.001);
        lastSpeed = measuredTokens / elapsed;

        if (
          now - lastRender >= RENDER_INTERVAL_MS ||
          generated.length === 1 ||
          generated.length === outputBudget
        ) {
          postMessage({
            type: "update",
            text: decoded,
            tokens: generated.length,
            speed: lastSpeed,
          });
          lastRender = now;
        }

        if (hasCompleteFinalBox(decoded)) {
          reason = "complete";
          break generation;
        }
        if (generated.length >= outputBudget) break generation;
      }

      const remaining = outputBudget - generated.length;
      pending = await activeEngine.decodeBatch(
        generated[generated.length - 1],
        Math.min(4, remaining),
      );
    }

    const text = decoder.finish();
    timings.totalMs = performance.now() - runStarted;
    postMessage({
      type: "done",
      text,
      tokens: generated.length,
      speed: lastSpeed,
      reason,
      timings,
    });
  } catch (error) {
    reportError(error);
  }
}

async function setup(timings: RunTimings): Promise<void> {
  if (engine && tokenizer) return;

  const tokenizerTask = tokenizer ? Promise.resolve(tokenizer) : loadTokenizer(timings);
  const engineTask = engine
    ? Promise.resolve(engine)
    : LimiteWebGpuEngine.create((progress) => {
        if (progress.phase === "manifest") {
          postStatus("preparing weights");
          return;
        }
        const ratio = progress.loadedBytes / progress.totalBytes;
        const action =
          progress.source === "cache" ? "reading cached weights" : "downloading weights";
        postStatus(
          `${action} · ${formatBytes(progress.loadedBytes)} / ${formatBytes(progress.totalBytes)}`,
          ratio,
        );
      });

  const [loadedTokenizer, loadedEngine] = await Promise.all([tokenizerTask, engineTask]);
  tokenizer = loadedTokenizer;
  engine = loadedEngine;
  timings.deviceMs = loadedEngine.timings.deviceMs;
  timings.artifactMs = loadedEngine.timings.artifactMs;
  timings.pipelineMs = loadedEngine.timings.pipelineMs;
}

async function loadTokenizer(timings: RunTimings): Promise<ViolettoTokenizer> {
  postStatus("preparing language");
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
  const system =
    "You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}.";
  return `<|endoftext|><|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${problem}<|im_end|>\n<|im_start|>assistant\n`;
}

function hasCompleteFinalBox(text: string): boolean {
  const thinkStart = text.indexOf("<think>");
  const thinkEnd = text.indexOf("</think>");
  if (thinkStart >= 0 && thinkEnd < thinkStart) return false;
  const answer = thinkEnd >= 0 ? text.slice(thinkEnd + "</think>".length) : text;
  const command = "\\boxed{";
  let start = answer.indexOf(command);
  while (start >= 0) {
    let depth = 1;
    for (let index = start + command.length; index < answer.length; index++) {
      if (answer[index] === "\\") {
        index += 1;
        continue;
      }
      if (answer[index] === "{") depth += 1;
      if (answer[index] === "}") depth -= 1;
      if (depth === 0) {
        const suffix = answer.slice(index + 1);
        const prefix = answer.slice(0, start);
        if (prefix.lastIndexOf("\\[") > prefix.lastIndexOf("\\]")) {
          return suffix.includes("\\]");
        }
        if (prefix.lastIndexOf("\\(") > prefix.lastIndexOf("\\)")) {
          return suffix.includes("\\)");
        }
        return true;
      }
    }
    start = answer.indexOf(command, start + command.length);
  }
  return false;
}

function reportError(error: unknown): void {
  console.error(error);
  postMessage({
    type: "error",
    message: error instanceof Error ? error.message : String(error),
  });
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
