import "@fontsource/courier-prime/400.css";
import { gsap } from "gsap";
import { Flip } from "gsap/Flip";
import katex from "katex";

import "./style.css";

gsap.registerPlugin(Flip);

type WorkerMessage =
  | { type: "status"; status: string; progress?: number }
  | {
      type: "ready";
      timings: {
        deviceMs?: number;
        tokenizerReadMs?: number;
        tokenizerParseMs?: number;
        modelReadMs?: number;
        modelUploadMs?: number;
      };
    }
  | {
      type: "update";
      text: string;
      tokens: number;
      speed: number;
      speeds: number[];
    }
  | {
      type: "done";
      text: string;
      tokens: number;
      speed: number;
      speeds: number[];
      reason: "complete" | "limit";
      timings: {
        prefillMs?: number;
        firstTokenMs?: number;
        totalMs?: number;
        sampler?: "wgsl" | "cpu";
        samplerGpuMs?: number;
        samplerCpuMs?: number;
      };
    }
  | { type: "diagnostic"; message: string }
  | { type: "error"; message: string };

const form = document.querySelector<HTMLFormElement>("#prompt-form")!;
const skipLink = document.querySelector<HTMLAnchorElement>(".skip-link")!;
const promptLine = document.querySelector<HTMLElement>(".prompt-line")!;
const prompt = document.querySelector<HTMLInputElement>("#prompt")!;
const run = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLElement>("#status")!;
const hint = document.querySelector<HTMLElement>("#hint")!;
const result = document.querySelector<HTMLElement>("#result")!;
const benchmarkOutput = document.querySelector<HTMLOutputElement>("#benchmark-output")!;
const answerSection = document.querySelector<HTMLElement>("#answer-section")!;
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
const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {
  type: "module",
});

let solving = false;
let finalRevealed = false;
let modelReady = false;
let introTimeline: gsap.core.Timeline | null = null;
let workingTimeline: gsap.core.Timeline | null = null;
let workTarget = "";
let workWritten = "";
let workComplete = false;
let workTimer: number | null = null;
let preparationStage = "";
let preparationProgressBucket = -1;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const benchmarkParams = new URLSearchParams(location.search);
const requestedHead = benchmarkParams.get("head");
const benchmarkHead =
  requestedHead === "wgsl" || requestedHead === "jax" ? requestedHead : "auto";
const benchmarkSeedValue = benchmarkParams.get("seed");
const benchmarkSeed = benchmarkSeedValue === null ? undefined : Number(benchmarkSeedValue);

console.info("[Kalkulator] preparing JAX.js and WebGPU");
startPreparationIntro();
worker.postMessage({ type: "prepare" });

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!modelReady || solving) return;

  const problem = prompt.value.trim();
  if (!problem) return;
  solving = true;
  form.classList.add("solving");
  prompt.disabled = true;
  run.disabled = true;
  resetOutput();
  startWorkingMotion();
  worker.postMessage({
    type: "solve",
    problem,
    options: {
      lmHead: benchmarkHead,
      ...(Number.isFinite(benchmarkSeed) ? { seed: benchmarkSeed } : {}),
    },
  });
});

worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === "ready") {
    modelReady = true;
    setStatus("ready");
    hint.textContent = "Ready";
    revealPrompt();
    console.info(`[Kalkulator] ready ${JSON.stringify(message.timings)}`);
    return;
  }

  if (message.type === "status") {
    logPreparation(message.status, message.progress);
    if (!modelReady) return;
    setStatus(message.status);
    if (message.progress !== undefined) {
      hint.textContent = `${Math.round(message.progress * 100)}%`;
    } else if (message.status === "reading cached model") {
      hint.textContent = "Reading local model";
    } else if (message.status === "uploading weights") {
      hint.textContent = "Model cached locally";
    } else if (message.status.startsWith("prefill")) {
      hint.textContent = "Model ready";
    }
    return;
  }

  if (message.type === "update") {
    const phase = renderOutput(message.text);
    setOutputPhase(phase);
    return;
  }

  if (message.type === "done") {
    const phase = renderOutput(message.text);
    if (phase !== "answer") {
      revealAnswerLayout();
      showAnswerMessage(
        message.reason === "limit"
          ? "No final answer before the output limit."
          : "No final answer.",
      );
    }
    answerSection.classList.remove("reasoning", "writing");
    const finalStatus = message.reason === "limit" ? "limit reached" : "complete";
    setStatus(finalStatus);
    hint.textContent = message.timings.prefillMs
      ? `Prefill ${(message.timings.prefillMs / 1000).toFixed(1)}s`
      : "Ready for another problem.";
    const summary = {
      tokens: message.tokens,
      tokensPerSecond: Number(message.speed.toFixed(2)),
      reason: message.reason,
      ...message.timings,
    };
    benchmarkOutput.textContent = `Benchmark ${JSON.stringify(summary)}`;
    console.info(`[Kalkulator] generation complete ${JSON.stringify(summary)}`);
    finishRun();
    return;
  }

  if (message.type === "diagnostic") {
    console.warn("[Kalkulator]", message.message);
    return;
  }

  console.error("[Kalkulator] preparation or inference failed", message.message);
  stopPreparationIntro();
  stopWorkingMotion();

  revealAnswerLayout();
  showAnswerMessage(message.message);
  setStatus("error");
  hint.textContent = "Try again or reload the page.";
  finishRun();
});

worker.addEventListener("error", (event) => {
  console.error("[Kalkulator] inference worker failed", event.error ?? event.message);
  stopPreparationIntro();
  stopWorkingMotion();
  revealAnswerLayout();
  showAnswerMessage(event.message || "Inference worker failed.");
  setStatus("error");
  finishRun();
});

function finishRun(): void {
  solving = false;
  form.classList.remove("solving");
  answerSection.classList.remove("reasoning", "writing");
  prompt.disabled = !modelReady;
  run.disabled = !modelReady;
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
  workComplete = false;
  answerSection.classList.add("waiting");
  answerSection.classList.remove("has-answer", "reasoning", "writing");
  workCopy.replaceChildren();
  finalCopy.replaceChildren();
  gsap.set([workCopy, finalCopy], { clearProps: "opacity,transform" });
  result.scrollTop = 0;
  if (artState) {
    Flip.from(artState, {
      absolute: true,
      duration: 0.56,
      ease: "power4.inOut",
    });
  }
}

function renderOutput(rawText: string): OutputPhase {
  const output = splitOutput(rawText);

  if (output.reasoning.trim()) {
    revealAnswerLayout();
    queueWorkText(output.reasoning, !output.thinking);
  }

  if (!output.answer.trim()) {
    return output.reasoning.trim() ? "reasoning" : "waiting";
  }

  flushWorkText(output.reasoning);
  revealAnswerLayout();
  const rendered = renderAnswerMath(output.answer);
  if (rendered && !finalRevealed) {
    finalRevealed = true;
    animateFinalIn();
  }
  return rendered ? "answer" : output.reasoning.trim() ? "reasoning" : "waiting";
}

function revealAnswerLayout(): void {
  if (!answerSection.classList.contains("waiting")) return;
  stopWorkingMotion();
  const artState = !reduceMotion ? Flip.getState(answerArt) : null;
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  if (artState) {
    Flip.from(artState, {
      absolute: true,
      duration: 0.68,
      ease: "power4.inOut",
    });
  }
}

function queueWorkText(value: string, complete: boolean): void {
  const text = cleanWorkText(value);
  if (!text.startsWith(workWritten)) {
    workWritten = "";
    workCopy.replaceChildren();
  }
  workTarget = text;
  workComplete = complete;
  if (workTimer === null) writeNextWorkWord();
}

function writeNextWorkWord(): void {
  workTimer = null;
  if (workWritten === workTarget) return;

  const remaining = workTarget.slice(workWritten.length);
  const next = remaining.match(
    workComplete ? /^(\s*\S+(?:\s+|$)|\s+$)/ : /^(\s*\S+\s+)/,
  )?.[0];
  if (!next) return;

  const firstWord = workWritten.length === 0;
  workWritten += next;
  workCopy.textContent = workWritten;
  if (firstWord && !reduceMotion) {
    gsap.fromTo(
      workCopy,
      { opacity: 0, y: 2 },
      { opacity: 1, y: 0, duration: 0.28, ease: "power3.out" },
    );
  }
  scrollToLatest();

  if (workWritten !== workTarget) {
    const pause = /[.!?;:]\s*$/.test(next) ? 72 : 34;
    workTimer = window.setTimeout(writeNextWorkWord, reduceMotion ? 0 : pause);
  }
}

function flushWorkText(value: string): void {
  stopWorkWriter();
  workTarget = cleanWorkText(value);
  workWritten = workTarget;
  workComplete = true;
  workCopy.textContent = workWritten;
  scrollToLatest();
}

function stopWorkWriter(): void {
  if (workTimer !== null) window.clearTimeout(workTimer);
  workTimer = null;
}

function cleanWorkText(value: string): string {
  return value.trimStart().replace(/\n{3,}/g, "\n\n");
}

function splitOutput(rawText: string): {
  reasoning: string;
  answer: string;
  thinking: boolean;
} {
  const trimmed = rawText.trimStart();
  if ("<think>".startsWith(trimmed) && trimmed !== "<think>") {
    return { reasoning: "", answer: "", thinking: true };
  }

  const start = rawText.indexOf("<think>");
  if (start < 0) return { reasoning: "", answer: rawText, thinking: false };
  const contentStart = start + "<think>".length;
  const end = rawText.indexOf("</think>", contentStart);
  if (end < 0) {
    return {
      reasoning: rawText.slice(contentStart).replace(/<\/?think[^>]*$/, ""),
      answer: "",
      thinking: true,
    };
  }
  return {
    reasoning: rawText.slice(contentStart, end),
    answer: rawText.slice(end + "</think>".length).trimStart(),
    thinking: false,
  };
}

function renderAnswerMath(rawText: string): boolean {
  finalCopy.replaceChildren();
  const text = stableMathPrefix(rawText)
    .replace(/\*\*(.*?)\*\*/gs, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n");
  const pattern = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    finalCopy.append(document.createTextNode(text.slice(cursor, index)));
    const raw = match[0];
    const display = raw.startsWith("$$") || raw.startsWith("\\[");
    const tex = raw.slice(
      display ? 2 : raw.startsWith("\\(") ? 2 : 1,
      display ? -2 : raw.startsWith("\\(") ? -2 : -1,
    );
    const node = document.createElement(display ? "div" : "span");
    katex.render(tex, node, {
      displayMode: display,
      throwOnError: false,
      strict: false,
    });
    finalCopy.append(node);
    cursor = index + raw.length;
  }
  finalCopy.append(document.createTextNode(text.slice(cursor)));
  scrollToLatest();
  return text.trim().length > 0;
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

function scrollToLatest(): void {
  const distance = result.scrollHeight - result.scrollTop - result.clientHeight;
  if (distance < 96) result.scrollTop = result.scrollHeight;
}

function stableMathPrefix(text: string): string {
  let end = text.length;
  const displayOpen = text.lastIndexOf("\\[");
  const displayClose = text.lastIndexOf("\\]");
  if (displayOpen > displayClose) end = Math.min(end, displayOpen);

  const doubleDollars = text.match(/\$\$/g)?.length ?? 0;
  if (doubleDollars % 2 === 1) end = Math.min(end, text.lastIndexOf("$$"));
  return text.slice(0, end);
}

function setStatus(value: string): void {
  status.textContent = value;
}
