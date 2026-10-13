import "@fontsource/courier-prime/400.css";
import { gsap } from "gsap";
import { Flip } from "gsap/Flip";
import katex from "katex";
import renderMathInElement from "katex/contrib/auto-render";

import { finalAnswerStart, findLastBoxedAnswer, hasCompleteFinalBox } from "./boxed-answer";
import { createFollowScroller } from "./follow-scroll";
import { mathFromClipboard } from "./math-paste";
import { LIMITE_SOLVE_TIMEOUT_MS, LIMITE_STALL_TIMEOUT_MS } from "./limite-config";
import {
  type CompletionReason,
  type InferenceRequest,
  type InferenceResponse,
} from "./inference-protocol";
import "./style.css";

gsap.registerPlugin(Flip);

const form = document.querySelector<HTMLFormElement>("#prompt-form")!;
const skipLink = document.querySelector<HTMLAnchorElement>(".skip-link")!;
const prompt = document.querySelector<HTMLInputElement>("#prompt")!;
const run = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLElement>("#status")!;
const hint = document.querySelector<HTMLElement>("#hint")!;
const activity = document.querySelector<HTMLElement>("#activity")!;
const answerSection = document.querySelector<HTMLElement>("#answer-section")!;
const copyFlow = document.querySelector<HTMLElement>("#copy-flow")!;
const answerArt = document.querySelector<HTMLElement>("#answer-art")!;
const answerArtGhost = document.querySelector<HTMLImageElement>(".answer-art-ghost")!;
const answerArtInk = document.querySelector<HTMLImageElement>(".answer-art-ink")!;
const workCopy = document.querySelector<HTMLElement>("#work-copy")!;
const finalCopy = document.querySelector<HTMLElement>("#final-copy")!;
const bloomShapes = [
  { x: 18, y: 18, at: 0 },
  { x: 82, y: 16, at: 0.72 },
  { x: 18, y: 76, at: 1.4 },
  { x: 77, y: 72, at: 2.02 },
  { x: 50, y: 48, at: 2.55 },
] as const;
const answerArtBlooms = bloomShapes.map((bloom) => {
  const layer = answerArtInk.cloneNode(false) as HTMLImageElement;
  layer.className = "answer-art-bloom";
  layer.style.setProperty("--bloom-x", `${bloom.x}%`);
  layer.style.setProperty("--bloom-y", `${bloom.y}%`);
  answerArt.insertBefore(layer, answerArtInk);
  return layer;
});
let worker: Worker | null = null;

let solving = false;
let nextRunId = 1;
let activeRunId: number | null = null;
let activeRawText = "";
type PausedRun = {
  readonly runId: number;
  readonly problem: string;
  readonly text: string;
  readonly elapsedMs: number;
};
let pausedRun: PausedRun | null = null;
let solveStarted = 0;
let lastProgressAt = 0;
let activityTimer: number | null = null;
let reloadRequired = false;
let finalRevealed = false;
let modelReady = false;
let introTimeline: gsap.core.Timeline | null = null;
let workingTimeline: gsap.core.Timeline | null = null;
let workTarget = "";
let workWritten = "";
let workTimer: number | null = null;
let workTextNode: Text | null = null;
let preparationStage = "";
let preparationProgressBucket = -1;
let generationProgressBucket = -1;
let preservedPastedProblem: { readonly raw: string; readonly display: string } | null = null;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const answerScroller = createFollowScroller(copyFlow);
console.info("[Kalkulator] preparing Violetto and WebGPU");
startPreparationIntro();
postWorkerMessage({ type: "prepare" });

prompt.addEventListener("paste", (event) => {
  const plain = event.clipboardData?.getData("text/plain") ?? "";
  const pasted = mathFromClipboard(event.clipboardData?.getData("text/html") ?? "") ?? plain;
  if (!pasted || (pasted === plain && !/[\r\n]/.test(pasted))) return;

  event.preventDefault();
  discardPausedRun();
  const start = prompt.selectionStart ?? prompt.value.length;
  const end = prompt.selectionEnd ?? start;
  const display = pasted.replace(/\s+/g, " ").trim();
  const replacesWholePrompt = start === 0 && end === prompt.value.length;
  prompt.setRangeText(display, start, end, "end");
  preservedPastedProblem = replacesWholePrompt
    ? { raw: pasted.trim(), display: prompt.value.trim() }
    : null;
});

prompt.addEventListener("input", () => {
  discardPausedRun();
  if (preservedPastedProblem?.display !== prompt.value.trim()) {
    preservedPastedProblem = null;
  }
});

function discardPausedRun(): void {
  if (!pausedRun) return;
  pausedRun = null;
  resetOutput();
  run.textContent = "solve";
  activity.textContent = "Ready";
  hint.textContent = "Violetto runs in your browser and can make mistakes.";
  setStatus("ready");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (reloadRequired) {
    window.location.reload();
    return;
  }
  if (!modelReady || solving) return;

  const displayedProblem = prompt.value.trim();
  if (!displayedProblem) return;
  const problem =
    preservedPastedProblem?.display === displayedProblem
      ? preservedPastedProblem.raw
      : displayedProblem;
  const saved = pausedRun?.problem === displayedProblem ? pausedRun : undefined;
  if (!saved) resetOutput();
  startModelSolve(problem, saved);
});

function startModelSolve(problem: string, saved?: PausedRun): void {
  const runId = saved?.runId ?? nextRunId++;
  solving = true;
  const segmentStarted = performance.now();
  solveStarted = segmentStarted - (saved?.elapsedMs ?? 0);
  lastProgressAt = segmentStarted;
  pausedRun = null;
  run.textContent = "solve";
  if (saved) {
    finalCopy.replaceChildren();
    finalRevealed = false;
  }
  activity.textContent = "Working";
  activity.title = `Up to ${formatDuration(LIMITE_SOLVE_TIMEOUT_MS)}. Press Escape to cancel.`;
  hint.textContent = `Violetto can make mistakes. Up to ${formatDuration(LIMITE_SOLVE_TIMEOUT_MS)}. Press Escape to cancel.`;
  activityTimer = window.setInterval(() => {
    activity.textContent = `Working ${formatDuration(performance.now() - solveStarted)}`;
    // The worker normally stops at its deadline. Recover if it cannot respond.
    if (!document.hidden && (
      performance.now() - lastProgressAt >= LIMITE_STALL_TIMEOUT_MS ||
      performance.now() - segmentStarted >= LIMITE_SOLVE_TIMEOUT_MS + 15_000
    )) {
      requireReload("The calculation stopped responding. Reload to try again.");
    }
  }, 1000);
  activeRunId = runId;
  activeRawText = saved?.text ?? "";
  answerSection.setAttribute("aria-busy", "true");
  form.classList.add("solving");
  prompt.disabled = true;
  run.disabled = true;
  preparationStage = "";
  preparationProgressBucket = -1;
  generationProgressBucket = -1;
  startWorkingMotion();
  const request: InferenceRequest = saved
    ? { type: "continue", runId }
    : { type: "solve", runId, problem };
  postWorkerMessage(request);
}

window.addEventListener("pagehide", () => {
  if (activeRunId !== null) {
    postWorkerMessage({ type: "cancel", runId: activeRunId });
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || event.isComposing || activeRunId === null) return;
  event.preventDefault();
  postWorkerMessage({ type: "cancel", runId: activeRunId });
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) lastProgressAt = performance.now();
});

function onInferenceMessage(event: MessageEvent<InferenceResponse>): void {
  if (reloadRequired) return;
  const message = event.data;
  if (message.type === "ready") {
    modelReady = true;
    setStatus("ready");
    hint.textContent = "Violetto runs in your browser and can make mistakes.";
    activity.textContent = "Ready";
    revealPrompt();
    console.info(`[Kalkulator] ready ${JSON.stringify(message.timings)}`);
    return;
  }

  if (message.type === "status") {
    if (message.runId !== undefined && message.runId !== activeRunId) return;
    if (message.runId === activeRunId) lastProgressAt = performance.now();
    logPreparation(message.status, message.progress);
    if (!modelReady) {
      const action = message.status.startsWith("downloading") ? "Downloading" : "Preparing";
      activity.textContent = message.progress === undefined
        ? "Preparing"
        : `${action} ${Math.round(message.progress * 100)}%`;
      return;
    }
    setStatus(message.status);
    return;
  }

  if (message.type === "update") {
    if (message.runId !== activeRunId) return;
    lastProgressAt = performance.now();
    logGeneration(message.tokens, message.speed);
    activeRawText += message.delta;
    const phase = renderOutput(activeRawText, false);
    setOutputPhase(phase);
    return;
  }

  if (message.type === "done" || message.type === "paused") {
    if (message.runId !== activeRunId) return;
    const paused = message.type === "paused";
    pausedRun = paused ? {
      runId: message.runId,
      problem: prompt.value.trim(),
      text: message.text,
      elapsedMs: message.timings.totalMs ?? performance.now() - solveStarted,
    } : null;
    activeRawText = message.text;
    const hasBoxedAnswer = !paused && message.reason === "eos" && hasCompleteFinalBox(message.text);
    renderOutput(message.text, hasBoxedAnswer);
    if (!hasBoxedAnswer) flushWorkText(splitOutput(message.text, false).reasoning);
    if (!hasBoxedAnswer) {
      revealAnswerLayout();
      showAnswerMessage(paused ? "No final answer yet. Continue from here." : failureMessage(message.reason));
    }
    answerSection.classList.remove("reasoning", "writing");
    const finalStatus = paused ? "paused" : hasBoxedAnswer ? "answer complete" : "no answer";
    setStatus(finalStatus);
    hint.textContent = paused
      ? "Continue the same calculation, or edit the problem to start a new one."
      : hasBoxedAnswer ? "Unverified model answer. Check it independently." : completionHint(message.reason);
    activity.textContent = paused
      ? `Paused at ${formatDuration(pausedRun!.elapsedMs)}`
      : hasBoxedAnswer ? `Finished in ${formatDuration(performance.now() - solveStarted)}` : completionHint(message.reason);
    const metrics = {
      tokens: message.tokens,
      tokensPerSecond: Number(message.speed.toFixed(2)),
      ...message.timings,
    };
    const summary = { runId: message.runId, reason: message.reason, ...metrics };
    console.info(`[Kalkulator] generation ${paused ? "paused" : "complete"} ${JSON.stringify(summary)}`);
    finishRun(message.runId);
    run.textContent = paused ? "continue" : "solve";
    return;
  }

  if (message.type === "rejected") {
    if (message.runId !== activeRunId) return;
    console.error("[Kalkulator] inference request rejected", message.message);
    stopWorkingMotion();
    revealAnswerLayout();
    showAnswerMessage(message.message);
    activity.textContent = "Could not finish";
    setStatus("error");
    hint.textContent = "Try again.";
    finishRun(message.runId);
    return;
  }

  if (message.runId !== undefined && message.runId !== activeRunId) return;
  console.error("[Kalkulator] preparation or inference failed", message.message);
  if (!modelReady || message.reloadRequired) {
    requireReload(message.message);
    return;
  }
  stopPreparationIntro();
  stopWorkingMotion();
  activity.textContent = "Could not finish";

  revealAnswerLayout();
  showAnswerMessage(message.message);
  setStatus("error");
  hint.textContent = "Try again or reload the page.";
  finishRun(message.runId);
}

function requireReload(message: string): void {
  pausedRun = null;
  worker?.terminate();
  worker = null;
  modelReady = false;
  reloadRequired = true;
  stopPreparationIntro();
  stopWorkingMotion();
  stopWorkWriter();
  finishRun(activeRunId ?? undefined);
  revealAnswerLayout();
  showAnswerMessage(message);
  setStatus("error");
  hint.textContent = "Reload to try again.";
  form.classList.remove("preparing");
  form.setAttribute("aria-hidden", "false");
  gsap.set(form, { autoAlpha: 1 });
  run.textContent = "reload";
  run.disabled = false;
  prompt.required = false;
  activity.textContent = "Reload to try again";
}

function finishRun(runId?: number): void {
  if (runId !== undefined && activeRunId !== runId) return;
  stopWorkWriter();
  stopWorkingMotion();
  if (activityTimer !== null) window.clearInterval(activityTimer);
  activityTimer = null;
  activity.title = "";
  solving = false;
  activeRunId = null;
  activeRawText = "";
  answerSection.setAttribute("aria-busy", "false");
  form.classList.remove("solving");
  answerSection.classList.remove("reasoning", "writing");
  prompt.disabled = !modelReady;
  run.disabled = !modelReady;
}

function failureMessage(reason: CompletionReason): string {
  switch (reason) {
    case "limit":
      return "No final answer within the available space. Try a simpler problem.";
    case "eos":
      return "Violetto stopped before returning a boxed final answer.";
    case "cancelled":
      return "The calculation was cancelled.";
    case "timeout":
      return `No final answer within ${formatDuration(LIMITE_SOLVE_TIMEOUT_MS)}. Try a simpler problem.`;
    case "repetition":
      return "The calculation began repeating and was stopped. Try rephrasing the problem.";
  }
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.floor(milliseconds / 1000);
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m${seconds % 60 ? ` ${seconds % 60}s` : ""}`;
}

function completionHint(reason: CompletionReason): string {
  switch (reason) {
    case "limit":
      return "Space limit reached";
    case "eos":
      return "Stopped before boxed answer";
    case "cancelled":
      return "Cancelled";
    case "timeout":
      return "Time limit reached";
    case "repetition":
      return "Repetition stopped";
  }
}

function postWorkerMessage(message: InferenceRequest): void {
  try {
    if (!worker) {
      if (message.type !== "prepare") return;
      worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module" });
      worker.addEventListener("message", onInferenceMessage);
      worker.addEventListener("error", (event) => {
        console.error("[Kalkulator] inference worker failed", event.error ?? event.message);
        requireReload("Violetto could not start. Reload to try again.");
      });
    }
    worker.postMessage(message);
  } catch {
    requireReload("This browser could not start Violetto. Reload to try again.");
  }
}

type OutputPhase = "waiting" | "reasoning" | "answer";

function setOutputPhase(phase: OutputPhase): void {
  answerSection.classList.toggle("reasoning", phase === "reasoning");
  answerSection.classList.toggle("writing", phase === "answer");
  setStatus(phase === "answer" ? "writing" : phase === "reasoning" ? "working" : "starting");
}

function logPreparation(value: string, progress?: number): void {
  const stage = value.split(" · ", 1)[0];
  if (progress !== undefined) {
    const bucket = Math.min(20, Math.floor(progress * 20));
    if (stage === preparationStage && bucket === preparationProgressBucket) return;
    preparationStage = stage;
    preparationProgressBucket = bucket;
    console.info(`[Kalkulator] ${value} · ${Math.round(progress * 100)}%`);
    return;
  }

  if (stage === preparationStage && preparationProgressBucket === -1) return;
  preparationStage = stage;
  preparationProgressBucket = -1;
  console.info(`[Kalkulator] ${value}`);
}

function logGeneration(tokens: number, speed: number): void {
  const bucket = Math.floor(tokens / 256);
  if (bucket === generationProgressBucket) return;
  generationProgressBucket = bucket;
  console.info(
    `[Kalkulator] generating · ${tokens} tok · ${speed.toFixed(2)} tok/s`,
  );
}

function startPreparationIntro(): void {
  gsap.set(answerArtGhost, { opacity: reduceMotion ? 0.16 : 0.08 });
  gsap.set(answerArtInk, { opacity: reduceMotion ? 0.18 : 0 });

  if (reduceMotion) return;
  gsap.from(answerArt, {
    opacity: 0,
    scale: 0.99,
    duration: 0.8,
    ease: "power3.out",
  });
  introTimeline = gsap.timeline({ repeat: -1, repeatDelay: 0.55 });
  bloomShapes.forEach((bloom, index) => {
    const layer = answerArtBlooms[index];
    introTimeline!
      .set(layer, { "--bloom-core": "0%", "--bloom-edge": "0%", opacity: 0 }, 0)
      .to(
        layer,
        { opacity: 0.96, duration: 0.65, ease: "power2.out" },
        bloom.at,
      )
      .to(
        layer,
        {
          "--bloom-core": "68%",
          "--bloom-edge": "100%",
          duration: 4.15,
          ease: "power1.inOut",
        },
        bloom.at,
      );
  });
  introTimeline
    .to(answerArtInk, { opacity: 1, duration: 0.9, ease: "power2.out" }, 6.15)
    .to(answerArtBlooms, { opacity: 0, duration: 0.45, ease: "power1.out" }, 6.35)
    .to(answerArtInk, { opacity: 0, duration: 1.3, ease: "power2.inOut" }, 8.05);
}

function stopPreparationIntro(): void {
  introTimeline?.kill();
  introTimeline = null;
  gsap.killTweensOf([answerArt, answerArtGhost, answerArtInk, ...answerArtBlooms]);
  gsap.to(answerArtInk, {
    opacity: 1,
    duration: reduceMotion ? 0.2 : 0.55,
    ease: "power3.out",
  });
  gsap.to(answerArtBlooms, {
    opacity: 0,
    duration: reduceMotion ? 0.2 : 0.24,
    ease: "power3.out",
  });
  gsap.to(answerArtGhost, {
    opacity: 0,
    duration: reduceMotion ? 0.2 : 0.36,
    ease: "power3.out",
  });
}

function startWorkingMotion(): void {
  workingTimeline?.kill();
  workingTimeline = null;
  gsap.killTweensOf([answerArtInk, ...answerArtBlooms]);
  gsap.set(answerArtGhost, { opacity: 0 });
  gsap.set(answerArtInk, { opacity: reduceMotion ? 1 : 0.72 });
  gsap.set(answerArtBlooms, {
    "--bloom-core": "18%",
    "--bloom-edge": "48%",
    opacity: 0,
  });

  if (reduceMotion) return;
  workingTimeline = gsap.timeline({ repeat: -1, repeatDelay: 0.08 });
  answerArtBlooms.forEach((layer, index) => {
    const at = index * 0.28;
    workingTimeline!
      .to(layer, { opacity: 0.34, duration: 0.52, ease: "power3.out" }, at)
      .to(layer, { opacity: 0, duration: 0.74, ease: "power2.inOut" }, at + 0.44);
  });
}

function stopWorkingMotion(): void {
  workingTimeline?.kill();
  workingTimeline = null;
  gsap.killTweensOf([answerArtInk, ...answerArtBlooms]);
  gsap.to(answerArtInk, {
    opacity: 1,
    duration: reduceMotion ? 0.2 : 0.32,
    ease: "power3.out",
  });
  gsap.to(answerArtBlooms, {
    opacity: 0,
    duration: reduceMotion ? 0.2 : 0.18,
    ease: "power3.out",
  });
}

function revealPrompt(): void {
  stopPreparationIntro();
  skipLink.classList.remove("preparing");
  skipLink.setAttribute("aria-hidden", "false");
  skipLink.tabIndex = 0;
  form.classList.remove("preparing");
  form.setAttribute("aria-hidden", "false");
  prompt.disabled = false;
  run.disabled = false;
  gsap.set(form, { autoAlpha: 1 });
  gsap.set([prompt, run], {
    opacity: 0,
    y: reduceMotion ? 0 : 3,
  });

  gsap
    .timeline({
      onComplete: () => {
        gsap.set([prompt, run], { clearProps: "opacity,transform" });
        prompt.focus({ preventScroll: true });
      },
    })
    .to(
      prompt,
      {
        opacity: 1,
        y: 0,
        duration: reduceMotion ? 0.2 : 0.42,
        ease: "power4.out",
      },
      0,
    )
    .to(
      run,
      {
        opacity: 0.34,
        y: 0,
        duration: reduceMotion ? 0.2 : 0.32,
        ease: "power4.out",
      },
      reduceMotion ? 0 : 0.08,
    );
}

function resetOutput(): void {
  stopWorkWriter();
  gsap.killTweensOf([answerArt, workCopy, finalCopy]);
  gsap.set(answerArt, { clearProps: "transform,opacity" });
  const artState =
    !reduceMotion && !answerSection.classList.contains("waiting")
      ? Flip.getState(answerArt)
      : null;
  finalRevealed = false;
  workTarget = "";
  workWritten = "";
  workTextNode = null;
  answerSection.classList.add("waiting");
  answerSection.classList.remove("has-answer", "reasoning", "writing");
  workCopy.replaceChildren();
  finalCopy.replaceChildren();
  answerScroller.reset();
  gsap.set([workCopy, finalCopy], { clearProps: "opacity,transform" });
  if (artState) {
    Flip.from(artState, {
      duration: 0.56,
      ease: "power4.inOut",
    });
  }
}

function renderOutput(rawText: string, complete: boolean): OutputPhase {
  const output = splitOutput(rawText, complete);

  if (output.reasoning.trim()) {
    revealAnswerLayout();
    if (!complete) queueWorkText(output.reasoning);
  }

  if (!output.answer.trim()) {
    return output.reasoning.trim() ? "reasoning" : "waiting";
  }

  flushWorkText(output.reasoning);
  // Typeset only after completion: streaming fragments stay stable and readable.
  // Single dollar signs remain currency, not guessed math delimiters.
  renderMathInElement(workCopy, {
    delimiters: [
      { left: "\\[", right: "\\]", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: false },
    ],
    throwOnError: false,
    strict: false,
    trust: false,
  });
  answerScroller.followLatest();
  revealAnswerLayout();
  katex.render(output.answer, finalCopy, {
    displayMode: false,
    throwOnError: false,
    strict: false,
  });
  if (!finalRevealed) {
    finalRevealed = true;
    animateFinalIn();
  }
  return "answer";
}

function revealAnswerLayout(): void {
  if (!answerSection.classList.contains("waiting")) return;
  stopWorkingMotion();
  const artState = !reduceMotion ? Flip.getState(answerArt) : null;
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  if (artState) {
    Flip.from(artState, {
      duration: 0.68,
      ease: "power4.inOut",
    });
  }
}

function queueWorkText(value: string): void {
  const text = cleanWorkText(value);
  if (!text.startsWith(workWritten)) {
    workWritten = "";
    workTextNode = null;
    workCopy.replaceChildren();
    answerScroller.reset();
  }
  workTarget = text;
  if (workTimer === null) writeNextWorkWord();
}

function writeNextWorkWord(): void {
  workTimer = null;
  if (workWritten === workTarget) return;

  const remaining = workTarget.slice(workWritten.length);
  // Keep the printed cadence without falling minutes behind a long response.
  const words = Math.min(8, Math.max(1, Math.ceil(remaining.length / 160)));
  const next = remaining.match(
    new RegExp(`^(?:\\s*\\S+\\s+){1,${words}}`),
  )?.[0];
  if (!next) return;

  const firstWord = workWritten.length === 0;
  workWritten += next;
  appendWorkText(next);
  if (firstWord && !reduceMotion) {
    gsap.fromTo(
      workCopy,
      { opacity: 0, y: 2 },
      { opacity: 1, y: 0, duration: 0.28, ease: "power3.out" },
    );
  }
  answerScroller.followLatest();

  if (workWritten !== workTarget) {
    const pause = words === 1 && /[.!?;:]\s*$/.test(next) ? 72 : 34;
    workTimer = window.setTimeout(writeNextWorkWord, reduceMotion ? 0 : pause);
  }
}

function flushWorkText(value: string): void {
  stopWorkWriter();
  workTarget = cleanWorkText(value);
  if (workTarget.startsWith(workWritten)) {
    appendWorkText(workTarget.slice(workWritten.length));
  } else {
    workCopy.replaceChildren();
    workTextNode = null;
    appendWorkText(workTarget);
  }
  workWritten = workTarget;
  answerScroller.followLatest();
}

function appendWorkText(value: string): void {
  if (!value) return;
  if (!workTextNode?.isConnected) {
    workTextNode = document.createTextNode("");
    workCopy.append(workTextNode);
  }
  workTextNode.appendData(value);
}

function stopWorkWriter(): void {
  if (workTimer !== null) window.clearTimeout(workTimer);
  workTimer = null;
}

function cleanWorkText(value: string): string {
  return value.trimStart().replace(/\s+/g, " ");
}

function splitOutput(rawText: string, complete: boolean): {
  reasoning: string;
  answer: string;
} {
  const trimmed = rawText.trimStart();
  if ("<think>".startsWith(trimmed) && trimmed !== "<think>") {
    return { reasoning: "", answer: "" };
  }

  const clean = (text: string) => text.replace(/<\/?think>/g, " ").replace(/<\/?think[^>]*$/, "");
  const box = complete ? findLastBoxedAnswer(rawText) : null;
  if (box && hasCompleteFinalBox(rawText)) {
    return {
      reasoning: clean(rawText.slice(0, finalAnswerStart(rawText, box)))
        .replace(/(?:\\\[|\\\(|\$\$?)\s*$/, ""),
      answer: rawText.slice(box.commandStart, box.closeBrace + 1),
    };
  }
  return {
    reasoning: clean(rawText),
    answer: "",
  };
}

function showAnswerMessage(message: string): void {
  finalCopy.replaceChildren(document.createTextNode(message));
  if (!finalRevealed) {
    finalRevealed = true;
    animateFinalIn();
  }
}

function animateFinalIn(): void {
  if (reduceMotion) return;
  gsap.fromTo(
    finalCopy,
    { opacity: 0, y: 2 },
    { opacity: 1, y: 0, duration: 0.36, ease: "power3.out" },
  );
}

function setStatus(value: string): void {
  if (status.textContent === value) return;
  status.textContent = value;
}
