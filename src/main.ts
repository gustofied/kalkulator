import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
import * as d3 from "d3";
import katex from "katex";

import { createState, loadModel, MODEL_URL, prefill, step, TOKENIZER_URL, type LimiteModel } from "./model";
import "./style.css";
import { ViolettoTokenizer } from "./tokenizer";

const form = document.querySelector<HTMLFormElement>("#prompt-form")!;
const prompt = document.querySelector<HTMLTextAreaElement>("#prompt")!;
const run = document.querySelector<HTMLButtonElement>("#run")!;
const status = document.querySelector<HTMLElement>("#status")!;
const answer = document.querySelector<HTMLElement>("#answer")!;
const trace = document.querySelector<SVGSVGElement>("#trace")!;

let model: LimiteModel | null = null;
let tokenizer: ViolettoTokenizer | null = null;
let initialized = false;

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void solve(prompt.value.trim());
});

async function solve(problem: string): Promise<void> {
  if (!problem || run.disabled) return;
  run.disabled = true;
  answer.classList.remove("muted");
  answer.replaceChildren();
  drawTrace([]);

  try {
    await setup();
    const activeModel = model!;
    const activeTokenizer = tokenizer!;
    const tokens = activeTokenizer.encode(formatPrompt(problem));
    const state = createState(tokens.length + 1024);
    const generated: number[] = [];
    const speeds: number[] = [];
    let logits: np.Array | null = null;
    const started = performance.now();

    try {
      setStatus(`prefill · ${tokens.length} tok`);
      logits = prefill(
        tree.ref(activeModel),
        np.array(tokens, { dtype: np.uint32 }),
        state,
      );

      for (let i = 0; i < 1024; i++) {
        const current = logits;
        logits = null;
        const next = await sample(current, 0.6, 50, 0.95);
        if (next === activeTokenizer.eosToken || next === activeTokenizer.imEndToken) break;

        generated.push(next);
        const elapsed = (performance.now() - started) / 1000;
        const speed = generated.length / Math.max(elapsed, 0.001);
        speeds.push(speed);
        renderMath(activeTokenizer.decode(generated));
        drawTrace(speeds);
        setStatus(`${generated.length} tok · ${speed.toFixed(1)} tok/s`);
        await new Promise(requestAnimationFrame);
        logits = step(tree.ref(activeModel), next, state);
      }

      if (generated.length === 0) answer.textContent = "End of text.";
      setStatus(`${generated.length} tok · done`);
    } finally {
      logits?.dispose();
      tree.dispose(state);
    }
  } catch (error) {
    console.error(error);
    answer.textContent = error instanceof Error ? error.message : String(error);
    setStatus("error");
  } finally {
    run.disabled = false;
  }
}

async function setup(): Promise<void> {
  if (!initialized) {
    setStatus("starting WebGPU");
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) throw new Error("WebGPU is unavailable in this browser.");
    defaultDevice("webgpu");
    initialized = true;
  }

  if (!tokenizer) {
    setStatus("loading tokenizer");
    tokenizer = ViolettoTokenizer.fromBinary(
      new Uint8Array(await fetchCached(TOKENIZER_URL)),
    );
  }

  if (!model) {
    setStatus("downloading 1.93 GiB");
    const checkpoint = await fetchCached(MODEL_URL);
    setStatus("uploading weights");
    model = await loadModel(checkpoint);
  }
}

async function fetchCached(url: string): Promise<ArrayBuffer> {
  const cache = await caches.open("kalkulator-v1");
  let response = await cache.match(url);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) throw new Error(`Download failed (${response.status}).`);
    try {
      await cache.put(url, response.clone());
    } catch (error) {
      console.warn("Checkpoint cache unavailable", error);
    }
  }
  return response.arrayBuffer();
}

function formatPrompt(problem: string): string {
  const system =
    "You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}.";
  return `<|endoftext|><|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${problem}<|im_end|>\n<|im_start|>assistant\n`;
}

async function sample(
  logits: np.Array,
  temperature: number,
  topK: number,
  topP: number,
): Promise<number> {
  const values = (await logits.data()) as Float32Array;
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

  const maximum = candidates[0].logit;
  const probabilities = candidates.map(({ logit }) =>
    Math.exp((logit - maximum) / temperature),
  );
  const total = d3.sum(probabilities);
  let keptTotal = 0;
  let kept = 0;
  while (kept < candidates.length) {
    keptTotal += probabilities[kept++];
    if (keptTotal / total >= topP) break;
  }
  let pick = Math.random() * keptTotal;
  for (let i = 0; i < kept; i++) {
    pick -= probabilities[i];
    if (pick <= 0) return candidates[i].id;
  }
  return candidates[kept - 1].id;
}

function renderMath(text: string): void {
  answer.replaceChildren();
  const thoughtEnd = text.lastIndexOf("</think>");
  if (thoughtEnd >= 0) text = text.slice(thoughtEnd + 8).trimStart();
  else if (text.includes("<think>")) return;
  text = text.replace(/\*\*(.*?)\*\*/gs, "$1").replace(/^#{1,6}\s+/gm, "");
  const pattern = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]+?\$)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    answer.append(document.createTextNode(text.slice(cursor, index)));
    const raw = match[0];
    const display = raw.startsWith("$$") || raw.startsWith("\\[");
    const tex = raw.slice(display ? 2 : raw.startsWith("\\(") ? 2 : 1, display ? -2 : raw.startsWith("\\(") ? -2 : -1);
    const node = document.createElement(display ? "div" : "span");
    katex.render(tex, node, { displayMode: display, throwOnError: false, strict: false });
    answer.append(node);
    cursor = index + raw.length;
  }
  answer.append(document.createTextNode(text.slice(cursor)));
}

function drawTrace(values: number[]): void {
  const width = trace.clientWidth || 600;
  const height = 48;
  const svg = d3.select(trace).attr("viewBox", `0 0 ${width} ${height}`);
  svg.selectAll("*").remove();
  if (values.length < 2) return;
  const x = d3.scaleLinear().domain([0, values.length - 1]).range([0, width]);
  const y = d3
    .scaleLinear()
    .domain([0, d3.max(values) ?? 1])
    .nice()
    .range([height - 3, 3]);
  const line = d3
    .line<number>()
    .x((_, i) => x(i))
    .y((d) => y(d));
  svg
    .append("path")
    .datum(values)
    .attr("fill", "none")
    .attr("stroke", "#5727e6")
    .attr("stroke-width", 1.5)
    .attr("d", line);
}

function setStatus(value: string): void {
  status.textContent = value;
}
