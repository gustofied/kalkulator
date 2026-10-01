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
      reason: "complete" | "limit" | "stopped";
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
const answerSection = document.querySelector<HTMLElement>("#answer-section")!;
const answerArt = document.querySelector<HTMLElement>("#answer-art")!;
const answerArtGhost = document.querySelector<HTMLImageElement>(".answer-art-ghost")!;
const answerArtInk = document.querySelector<HTMLImageElement>(".answer-art-ink")!;
const answerCopy = document.querySelector<HTMLElement>("#answer-copy")!;
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
let answerRevealed = false;
let modelReady = false;
let introTimeline: gsap.core.Timeline | null = null;
let preparationStage = "";
let preparationProgressBucket = -1;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

console.info("[Kalkulator] preparing JAX.js and WebGPU");
startPreparationIntro();
worker.postMessage({ type: "prepare" });

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!modelReady) return;
  if (solving) {
    run.disabled = true;
    run.textContent = "Stopping";
    setStatus("stopping");
    worker.postMessage({ type: "stop" });
    return;
  }

  const problem = prompt.value.trim();
  if (!problem) return;
  solving = true;
  run.textContent = "Stop";
  run.dataset.mode = "stop";
  resetOutput();
  worker.postMessage({ type: "solve", problem });
});

worker.addEventListener("message", (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === "ready") {
    modelReady = true;
    setStatus("ready");
    hint.textContent = "Ready";
    revealPrompt();
    console.info("[Kalkulator] ready", message.timings);
    return;
  }

  if (message.type === "status") {
    logPreparation(message.status, message.progress);
    if (!modelReady) return;
    setStatus(message.status);
    if (message.progress !== undefined) {
      hint.textContent = `${Math.round(message.progress * 100)}%`;
    } else if (message.status === "reading cached model") {
      hint.textContent = "Reading the local model…";
    } else if (message.status === "uploading weights") {
      hint.textContent = "Model cached locally";
    } else if (message.status.startsWith("prefill")) {
      hint.textContent = "Model ready";
    }
    return;
  }

  if (message.type === "update") {
    const hasAnswer = renderOutput(message.text);
    answerSection.classList.toggle("writing", hasAnswer);
    setStatus(hasAnswer ? "writing" : "thinking");
    return;
  }

  if (message.type === "done") {
    const hasAnswer = renderOutput(message.text);
    if (!hasAnswer) {
      answerSection.classList.remove("waiting");
      answerSection.classList.add("has-answer");
      showAnswerMessage(
        message.reason === "stopped"
          ? "Stopped before a final answer."
          : message.reason === "limit"
            ? "The model reached its output limit before returning a final answer."
            : "The model ended before returning a final answer.",
      );
    }
    answerSection.classList.remove("writing");
    const finalStatus =
      message.reason === "stopped"
        ? "stopped"
        : message.reason === "limit"
          ? "limit reached"
          : "complete";
    setStatus(finalStatus);
    hint.textContent = message.timings.prefillMs
      ? `Prefill ${(message.timings.prefillMs / 1000).toFixed(1)}s`
      : "Ready for another problem.";
    console.info("[Kalkulator] generation complete", message.timings);
    finishRun();
    return;
  }

  if (message.type === "diagnostic") {
    console.warn("[Kalkulator]", message.message);
    return;
  }

  console.error("[Kalkulator] preparation or inference failed", message.message);
  stopPreparationIntro();

  const artState =
    !reduceMotion && answerSection.classList.contains("waiting")
      ? Flip.getState(answerArt)
      : null;
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  if (artState) {
    Flip.from(artState, {
      absolute: true,
      duration: 0.9,
      ease: "power3.inOut",
    });
  }
  showAnswerMessage(message.message);
  setStatus("error");
  hint.textContent = "Try again or reload the page.";
  finishRun();
});

worker.addEventListener("error", (event) => {
  console.error("[Kalkulator] inference worker failed", event.error ?? event.message);
  stopPreparationIntro();
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  showAnswerMessage(event.message || "Inference worker failed.");
  setStatus("error");
  finishRun();
});

function finishRun(): void {
  solving = false;
  answerSection.classList.remove("writing");
  run.disabled = !modelReady;
  run.textContent = "ask";
  delete run.dataset.mode;
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
    scale: 0.98,
    duration: 1.1,
    ease: "power2.out",
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
    duration: reduceMotion ? 0.2 : 0.7,
    ease: "power2.out",
  });
  gsap.to(answerArtBlooms, {
    opacity: 0,
    duration: reduceMotion ? 0.2 : 0.45,
    ease: "power2.out",
  });
  gsap.to(answerArtGhost, {
    opacity: 0,
    duration: reduceMotion ? 0.2 : 0.5,
    ease: "power2.out",
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
  gsap.set(promptLine, { "--line-scale": reduceMotion ? 1 : 0 });
  gsap.set([prompt, run], {
    opacity: 0,
    y: reduceMotion ? 0 : 5,
  });

  gsap
    .timeline({
      onComplete: () => {
        gsap.set([prompt, run], { clearProps: "opacity,transform" });
        prompt.focus({ preventScroll: true });
      },
    })
    .to(promptLine, {
      "--line-scale": 1,
      duration: reduceMotion ? 0.2 : 0.9,
      ease: "power3.inOut",
    })
    .to(
      prompt,
      {
        opacity: 1,
        y: 0,
        duration: reduceMotion ? 0.2 : 0.65,
        ease: "power3.out",
      },
      reduceMotion ? 0 : 0.18,
    )
    .to(
      run,
      {
        opacity: 0.34,
        y: 0,
        duration: reduceMotion ? 0.2 : 0.55,
        ease: "power3.out",
      },
      reduceMotion ? 0 : 0.32,
    );
}

function resetOutput(): void {
  answerRevealed = false;
  answerSection.classList.add("waiting");
  answerSection.classList.remove("has-answer", "writing");
  answerCopy.replaceChildren();
  result.scrollTop = 0;
  gsap.set(answerArt, { clearProps: "transform,opacity" });
}

function renderOutput(rawText: string): boolean {
  const output = splitOutput(rawText);

  if (!output.answer.trim()) {
    answerSection.classList.add("waiting");
    answerSection.classList.remove("has-answer");
    answerCopy.replaceChildren();
    return false;
  }

  const artState =
    !reduceMotion && answerSection.classList.contains("waiting")
      ? Flip.getState(answerArt)
      : null;
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  if (artState) {
    Flip.from(artState, {
      absolute: true,
      duration: 0.9,
      ease: "power3.inOut",
    });
  }
  const rendered = renderAnswerMath(output.answer);
  if (rendered && !answerRevealed) {
    answerRevealed = true;
    animateAnswerIn();
  }
  return rendered;
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
  answerCopy.replaceChildren();
  const text = stableMathPrefix(rawText)
    .replace(/\*\*(.*?)\*\*/gs, "$1")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\n{3,}/g, "\n\n");
  const pattern = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    answerCopy.append(document.createTextNode(text.slice(cursor, index)));
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
    answerCopy.append(node);
    cursor = index + raw.length;
  }
  answerCopy.append(document.createTextNode(text.slice(cursor)));
  result.scrollTop = result.scrollHeight;
  return text.trim().length > 0;
}

function showAnswerMessage(message: string): void {
  answerCopy.replaceChildren(document.createTextNode(message));
  if (!answerRevealed) {
    answerRevealed = true;
    animateAnswerIn();
  }
}

function animateAnswerIn(): void {
  if (reduceMotion) return;
  gsap.fromTo(
    answerCopy,
    { opacity: 0 },
    { opacity: 1, duration: 0.8, ease: "power2.out" },
  );
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
