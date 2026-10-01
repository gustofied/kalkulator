import "@fontsource/courier-prime/400.css";
import "@fontsource/courier-prime/400-italic.css";
import { gsap } from "gsap";
import katex from "katex";

import "./style.css";

type WorkerMessage =
  | { type: "status"; status: string; progress?: number }
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
const prompt = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const run = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLElement>("#status")!;
const hint = document.querySelector<HTMLElement>("#hint")!;
const result = document.querySelector<HTMLElement>("#result")!;
const answerSection = document.querySelector<HTMLElement>("#answer-section")!;
const answerArt = document.querySelector<HTMLElement>("#answer-art")!;
const answerCopy = document.querySelector<HTMLElement>("#answer-copy")!;
const worker = new Worker(new URL("./inference.worker.ts", import.meta.url), {
  type: "module",
});

let solving = false;
let answerRevealed = false;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

if (!reduceMotion) {
  gsap
    .timeline({ defaults: { duration: 1.1, ease: "power2.out" } })
    .from(".site-header", { opacity: 0 })
    .from("#prompt-form", { opacity: 0 }, "-=0.75")
    .from("#result", { opacity: 0 }, "-=0.85")
    .from(".site-footer", { opacity: 0 }, "-=0.95");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
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
  if (message.type === "status") {
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
    console.info("Kalkulator timings", JSON.stringify(message.timings));
    finishRun();
    return;
  }

  if (message.type === "diagnostic") {
    console.warn(message.message);
    return;
  }

  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  showAnswerMessage(message.message);
  setStatus("error");
  hint.textContent = "Try again or reload the page.";
  finishRun();
});

worker.addEventListener("error", (event) => {
  console.error(event.error ?? event.message);
  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
  showAnswerMessage(event.message || "Inference worker failed.");
  setStatus("error");
  finishRun();
});

function finishRun(): void {
  solving = false;
  answerSection.classList.remove("writing");
  run.disabled = false;
  run.textContent = "ask";
  delete run.dataset.mode;
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

  answerSection.classList.remove("waiting");
  answerSection.classList.add("has-answer");
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
    answerArt,
    { opacity: 0 },
    { opacity: 1, duration: 1.1, ease: "power2.out" },
  );
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
