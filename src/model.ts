// Violetto's architecture, expressed as JAX.js array operations.
import { blockUntilReady, jit, lax, nn, numpy as np, tree } from "@jax-js/jax";
import { q4Embedding, q4Linear, type Q4Linear } from "./jax-q4";
import { readLimiteQ4Shards, type LimiteArtifactProgressCallback } from "./model-artifact";

const C = {
  vocab: 151_680,
  hidden: 1_280,
  intermediate: 3_328,
  layers: 48,
  heads: 10,
  kvHeads: 2,
  groups: 5,
  headDim: 128,
  slidingWindow: 1_024,
  attentionScale: 0.1,
  rmsEps: 1 / 128,
  xsaEps: 1e-4,
  cacheBlock: 256,
} as const;

type Attention = {
  qkvProj: Q4Linear;
  oProj: Q4Linear;
  xsaFactor: np.Array;
};
type MLP = {
  gateUpProj: Q4Linear;
  downProj: Q4Linear;
};
type Layer = {
  selfAttn: Attention;
  mlp: MLP;
  residLambdaAttn: np.Array;
  postLambdaAttn: np.Array;
  residLambdaMlp: np.Array;
  postLambdaMlp: np.Array;
};
type Mudd = {
  dense1: np.Array;
  dense2: np.Array;
  dense2Mlp: np.Array;
  bias: np.Array;
  biasMlp: np.Array;
};
export type LimiteModel = {
  embedTokens: Q4Linear;
  valueEmbeds: Q4Linear;
  layers: Layer[];
  mudd: Mudd;
};
type KV = { data: np.Array };
export type LimiteState = { caches: KV[]; position: number; capacity: number };

const GLOBAL = new Set([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
const VALUE = new Set([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]);

function rmsNorm(x: np.Array): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const meanSquare = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  return x.div(np.sqrt(meanSquare.add(C.rmsEps))).astype(dtype);
}

function projectAttention(
  attn: Attention,
  hidden: np.Array,
): {
  queryKey: np.Array;
  value: np.Array;
  attentionGate: np.Array;
  valueGate?: np.Array;
} {
  const T = hidden.shape[0];
  const projected = q4Linear(attn.qkvProj, hidden);
  const qEnd = C.heads * C.headDim;
  const kEnd = qEnd + C.kvHeads * C.headDim;
  const vEnd = kEnd + C.kvHeads * C.headDim;
  const gateEnd = vEnd + C.heads;
  // The artifact has already folded QKV/O scales into the packed weights.
  // Q and K have the same per-head normalization. Keep them together so JAX
  // can normalize and rotate both in one set of GPU operations.
  const queryKey = rmsNorm(projected.ref.slice([], [0, kEnd])
    .reshape([T, C.heads + C.kvHeads, C.headDim]));
  const value = projected.ref.slice([], [kEnd, vEnd]);
  const valueGate = projected.shape[1] > gateEnd ? projected.ref.slice([], [gateEnd]) : undefined;
  const attentionGate = projected.slice([], [vEnd, gateEnd]);
  return {
    queryKey,
    value,
    attentionGate,
    valueGate,
  };
}

/** Round finite FP32 values to BF16, retaining FP32 storage for WebGPU. */
function roundBfloat16(values: np.Array): np.Array {
  const bits = lax.bitcastConvertType(values, np.uint32);
  const tiesToEven = np.bitwiseAnd(
    np.rightShift(bits.ref, np.array(16, { dtype: np.uint32 })),
    np.array(1, { dtype: np.uint32 }),
  );
  const rounded = bits.add(tiesToEven.add(np.array(0x7fff, { dtype: np.uint32 })));
  return lax.bitcastConvertType(
    np.bitwiseAnd(rounded, np.array(0xffff0000, { dtype: np.uint32 })), np.float32,
  );
}

const makeRoPEFactors = jit(
  function makeRoPEFactors(offset: np.Array, length: number): [np.Array, np.Array] {
    const base = np.exp(np.linspace(0, 1, 32).mul(-Math.log(1024)));
    const frequency = np.concatenate([
      np.repeat(base, 2),
      np.zeros([C.headDim - 64], { dtype: np.float32 }),
    ]);
    const positions = np
      .arange(length, undefined, undefined, { dtype: np.float32 })
      .add(offset)
      .reshape([length, 1]);
    const theta = positions.mul(frequency.reshape([1, C.headDim]));
    // The checkpoint specifies BF16 trigonometric factors even when the
    // surrounding attention arithmetic uses FP32.
    const cosine = roundBfloat16(np.cos(theta.ref)).reshape([length, 1, C.headDim]);
    let sine = roundBfloat16(np.sin(theta));
    const odd = np
      .remainder(np.arange(C.headDim), 2)
      .equal(1)
      .reshape([1, C.headDim]);
    sine = np
      .where(odd, sine.ref.mul(-1), sine)
      .reshape([length, 1, C.headDim]);
    return [cosine, sine];
  },
  { staticArgnums: [1] },
);

function applyRoPE(
  queryKey: np.Array,
  cosine: np.Array,
  sine: np.Array,
): np.Array {
  const [T, heads, D] = queryKey.shape;
  const paired = np.flip(queryKey.ref.reshape([T, heads, D / 2, 2]), -1)
    .reshape([T, heads, D]);
  return queryKey.mul(cosine).add(paired.mul(sine));
}

function applyValueEmbedding(
  gateLogits: np.Array,
  valueEmbeds: np.Array,
  values: np.Array,
): np.Array {
  const gate = nn.sigmoid(gateLogits).mul(2);
  return values.add(gate.reshape([values.shape[0], C.kvHeads, 1]).mul(valueEmbeds));
}

function applyXsa(attn: Attention, output: np.Array, currentValues: np.Array): np.Array {
  const T = output.shape[0];
  const grouped = output.reshape([T, C.kvHeads, C.groups, C.headDim]);
  const norm = np.sqrt(
    currentValues.ref.mul(currentValues.ref).sum(-1, { keepdims: true }),
  );
  const direction = currentValues
    .div(np.maximum(norm, C.xsaEps))
    .reshape([T, C.kvHeads, 1, C.headDim]);
  const projection = grouped.ref
    .mul(direction.ref)
    .sum(-1, { keepdims: true });
  const alpha = attn.xsaFactor.reshape([1, C.kvHeads, C.groups, 1]);
  return grouped
    .sub(alpha.mul(projection).mul(direction))
    .reshape([T, C.heads, C.headDim]);
}

function applyAttentionGate(gateLogits: np.Array, output: np.Array): np.Array {
  const gate = nn.sigmoid(gateLogits).mul(2);
  return output.mul(gate.reshape([output.shape[0], C.heads, 1]));
}

function attentionPrefill(
  attn: Attention,
  hidden: np.Array,
  valueEmbeds: np.Array,
  ropeCosine: np.Array,
  ropeSine: np.Array,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = hidden.shape[0];
  const projected = projectAttention(attn, hidden);
  let { queryKey, value: v } = projected;
  v = v.reshape([
    T,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) {
    v = applyValueEmbedding(projected.valueGate!, valueEmbeds, v);
  } else {
    valueEmbeds.dispose();
  }
  const currentValues = v.ref;
  if (!isGlobal) queryKey = applyRoPE(queryKey, ropeCosine, ropeSine);
  else { ropeCosine.dispose(); ropeSine.dispose(); }
  const q = queryKey.ref.slice([], [0, C.heads]);
  const k = queryKey.slice([], [C.heads]);

  let output = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: C.attentionScale,
    localWindowSize: isGlobal ? undefined : [C.slidingWindow, 0],
  });
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(projected.attentionGate, output);
  output = q4Linear(
    attn.oProj,
    output.reshape([T, C.heads * C.headDim]),
  );
  return { output, key: k, value: v };
}

function attentionStep(
  attn: Attention,
  cache: KV,
  hidden: np.Array,
  valueEmbeds: np.Array,
  ropeCosine: np.Array,
  ropeSine: np.Array,
  slot: np.Array,
  validLength: np.Array,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): { output: np.Array; cache: KV } {
  const projected = projectAttention(attn, hidden);
  let { queryKey, value: v } = projected;
  v = v.reshape([
    1,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) {
    v = applyValueEmbedding(projected.valueGate!, valueEmbeds, v);
  } else {
    valueEmbeds.dispose();
  }
  const currentValues = v.ref;
  if (!isGlobal) queryKey = applyRoPE(queryKey, ropeCosine, ropeSine);
  else { ropeCosine.dispose(); ropeSine.dispose(); }
  const q = queryKey.ref.slice([], [0, C.heads]);
  const k = queryKey.slice([], [C.heads]);

  const capacity = cache.data.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1, 1]);
  const update = np.stack([k, v], 1).astype(np.float16);
  const data = np.where(
    slotMask,
    np.tile(update, [capacity, 1, 1, 1]),
    cache.data,
  );
  let output = decodeAttention(q, data.ref, validLength);
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(projected.attentionGate, output);
  output = q4Linear(
    attn.oProj,
    output.reshape([1, C.heads * C.headDim]),
  );
  return { output, cache: { data } };
}

/** Grouped attention with parallel, bounded-length cache reductions. */
function decodeAttention(query: np.Array, data: np.Array, validLength: np.Array): np.Array {
  const length = data.shape[0];
  // Keep cache views in FP16 through indexing and padding. The FP32 query and
  // probabilities promote the products, without expanding the entire cache.
  const key = data.ref.slice([], 0);
  const value = data.slice([], 1);
  // A single query has only ten heads. Small cache tiles expose enough
  // independent reductions to occupy the GPU; longer global caches use
  // larger tiles to keep the final reduction and scratch space bounded.
  const blockSize = length <= C.slidingWindow + 1 ? 16 : 128;
  const padded = Math.ceil(length / blockSize) * blockSize;
  const blocks = padded / blockSize;
  const scores = np.einsum(
    "kgd,skd->kgs", query.reshape([C.kvHeads, C.groups, C.headDim]), key,
  ).mul(C.attentionScale);
  // Mask after padding: padded slots must never contribute to softmax.
  const masked = np.where(
    np.arange(padded).less(validLength),
    np.pad(scores, { 2: [0, padded - length] }),
    -Infinity,
  ).reshape([C.kvHeads, C.groups, blocks, blockSize]);
  const maximum = masked.ref.max(-1, { keepdims: true }).max(-2, { keepdims: true });
  const masses = np.exp(masked.sub(maximum));
  const total = masses.ref.sum(-1, { keepdims: true }).sum(-2, { keepdims: true });
  const probabilities = masses.div(total);
  const values = np.pad(value, { 0: [0, padded - length] })
    .reshape([blocks, blockSize, C.kvHeads, C.headDim]).transpose([2, 0, 1, 3]);
  return probabilities.reshape([C.kvHeads, C.groups, blocks, blockSize, 1])
    .mul(values.reshape([C.kvHeads, 1, blocks, blockSize, C.headDim]))
    .sum(-2).sum(-2).reshape([1, C.heads, C.headDim]);
}

function runMlp(mlp: MLP, hidden: np.Array): np.Array {
  const gateUp = q4Linear(mlp.gateUpProj, hidden);
  const gate = nn.silu(gateUp.ref.slice([], [0, C.intermediate]));
  const up = gateUp.slice([], [C.intermediate]);
  return q4Linear(mlp.downProj, gate.mul(up));
}

function padCache(data: np.Array, capacity: number): KV {
  const T = data.shape[0];
  if (T > capacity) throw new Error("Prompt exceeds context capacity");
  if (T === capacity) return { data };
  return {
    data: np.pad(data, { 0: [0, capacity - T] }),
  };
}

const runLayerPrefill = jit(
  function runLayerPrefill(
    layer: Layer,
    attentionInput: np.Array,
    residualBase: np.Array,
    valueEmbeds: np.Array,
    ropeCosine: np.Array,
    ropeSine: np.Array,
    isGlobal: boolean,
    hasValueEmbedding: boolean,
    capacity: number,
  ): [np.Array, KV] {
    const { output, key, value } = attentionPrefill(
      layer.selfAttn,
      attentionInput,
      valueEmbeds,
      ropeCosine,
      ropeSine,
      isGlobal,
      hasValueEmbedding,
    );
    const mixed = residualBase
      .mul(layer.residLambdaAttn)
      .add(output.mul(layer.postLambdaAttn));
    const mlp = runMlp(layer.mlp, rmsNorm(mixed.ref));
    const result = mixed
      .mul(layer.residLambdaMlp)
      .add(mlp.mul(layer.postLambdaMlp));
    let data = np.stack([key, value], 1).astype(np.float16);
    if (!isGlobal && data.shape[0] > C.slidingWindow + 1) {
      const count = C.slidingWindow + 1;
      const first = data.shape[0] - count;
      const slots = np.remainder(np.arange(count).sub(first % count).add(count), count).add(first);
      data = np.take(data, slots, 0);
    }
    return [result, padCache(data, isGlobal ? capacity : Math.min(capacity, C.slidingWindow + 1))];
  },
  { staticArgnums: [6, 7, 8] },
);

function runLayerStep(
  layer: Layer,
  cache: KV,
  attentionInput: np.Array,
  residualBase: np.Array,
  valueEmbeds: np.Array,
  ropeCosine: np.Array,
  ropeSine: np.Array,
  slot: np.Array,
  validLength: np.Array,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): [np.Array, KV] {
  const { output, cache: nextCache } = attentionStep(
    layer.selfAttn,
    cache,
    attentionInput,
    valueEmbeds,
    ropeCosine,
    ropeSine,
    slot,
    validLength,
    isGlobal,
    hasValueEmbedding,
  );
  const mixed = residualBase
    .mul(layer.residLambdaAttn)
    .add(output.mul(layer.postLambdaAttn));
  const mlp = runMlp(layer.mlp, rmsNorm(mixed.ref));
  const result = mixed
    .mul(layer.residLambdaMlp)
    .add(mlp.mul(layer.postLambdaMlp));
  return [result, nextCache];
}

function runMuddPair(
  mudd: Mudd,
  tap0: np.Array,
  tap1: np.Array,
  tap2: np.Array,
  current: np.Array,
  layerIndex: number,
): [np.Array, np.Array] {
  const T = current.shape[0];
  const inner = nn.gelu(
    np.dot(rmsNorm(current), mudd.dense1.transpose()),
    { approximate: false },
  );
  const makeMix = (
    dense2: np.Array,
    bias: np.Array,
    retainInputs: boolean,
  ): np.Array => {
    const weights = np
      .dot(
        retainInputs ? inner.ref : inner,
        dense2.slice(layerIndex).transpose(),
      )
      .add(bias.slice(layerIndex));
    const i0 = np.array(0, { dtype: np.uint32 });
    const i1 = np.array(1, { dtype: np.uint32 });
    const i2 = np.array(2, { dtype: np.uint32 });
    const a = retainInputs ? tap0.ref : tap0;
    const b = retainInputs ? tap1.ref : tap1;
    const c = retainInputs ? tap2.ref : tap2;
    return a
      .mul(np.take(weights.ref, i0, -1).reshape([T, 1]))
      .add(b.mul(np.take(weights.ref, i1, -1).reshape([T, 1])))
      .add(c.mul(np.take(weights, i2, -1).reshape([T, 1])));
  };
  return [
    makeMix(mudd.dense2, mudd.bias, true),
    makeMix(mudd.dense2Mlp, mudd.biasMlp, false),
  ];
}

function roundCapacity(required: number): number {
  return Math.max(C.cacheBlock, Math.ceil(required / C.cacheBlock) * C.cacheBlock);
}

export function createState(capacity: number = C.cacheBlock): LimiteState {
  capacity = roundCapacity(capacity);
  return {
    capacity,
    position: 0,
    // Prefill creates every cache directly. Avoid allocating and immediately
    // disposing a capacity-sized zero cache for all 48 layers.
    caches: [],
  };
}

function ensureCapacity(state: LimiteState, required: number): void {
  if (state.capacity >= required) return;
  const old = state.capacity;
  const next = roundCapacity(Math.max(required, old * 2));
  for (let i = 0; i < state.caches.length; i++) {
    // Grow local caches with the prefix, then retain the full logical ring.
    // Before the first wrap, this avoids scanning and copying empty slots.
    const capacity = GLOBAL.has(i) ? next : Math.min(next, C.slidingWindow + 1);
    const cache = state.caches[i];
    if (cache.data.shape[0] < capacity) {
      cache.data = np.pad(cache.data, { 0: [0, capacity - cache.data.shape[0]] });
    }
  }
  state.capacity = next;
}

function layerInputs(
  model: LimiteModel,
  hidden: np.Array,
  layerIndex: number,
  history0: np.Array,
  history12: np.Array | null,
  history23: np.Array | null,
): [np.Array, np.Array] {
  if (layerIndex === 24) {
    if (!history12) throw new Error("Missing MUDD tap 12");
    const [attentionMix, residualMix] = runMuddPair(
      tree.ref(model.mudd),
      history0.ref,
      history12.ref,
      hidden.ref,
      hidden.ref,
      layerIndex,
    );
    return [rmsNorm(attentionMix), residualMix];
  }
  if (layerIndex === 47) {
    if (!history23) throw new Error("Missing MUDD tap 23");
    const [attentionMix, residualMix] = runMuddPair(
      tree.ref(model.mudd),
      history0.ref,
      history23.ref,
      hidden.ref,
      hidden.ref,
      layerIndex,
    );
    return [rmsNorm(attentionMix), residualMix];
  }
  return [rmsNorm(hidden.ref), hidden.ref];
}

const runHead = jit((linear: Q4Linear, hidden: np.Array): np.Array =>
  q4Linear(linear, hidden).reshape([C.vocab]),
);

export function logitsFromHidden(model: LimiteModel, hidden: np.Array): np.Array {
  // Return raw logits; the sampler must softcap the complete valid vocabulary.
  return runHead(tree.ref(model.embedTokens), hidden);
}

export function prefill(
  model: LimiteModel,
  tokenIds: np.Array,
  state: LimiteState,
): np.Array {
  const promptLength = tokenIds.shape[0];
  // Reuse compiled prefill shapes between short problems. Causal padding never
  // enters the valid cache prefix, and decode overwrites it as it advances.
  const sequenceLength = promptLength <= 256
    ? 2 ** Math.ceil(Math.log2(promptLength)) : promptLength;
  if (sequenceLength !== promptLength) {
    tokenIds = np.pad(tokenIds, [[0, sequenceLength - promptLength]]);
  }
  ensureCapacity(state, sequenceLength);
  let hidden = q4Embedding(tree.ref(model.embedTokens), tokenIds.ref);
  hidden = rmsNorm(hidden);
  const valueEmbeds = q4Embedding(tree.ref(model.valueEmbeds), tokenIds)
    .reshape([hidden.shape[0], C.kvHeads, C.headDim]);
  const [ropeCosine, ropeSine] = makeRoPEFactors(
    np.array(0, { dtype: np.uint32 }),
    sequenceLength,
  );
  const history0 = hidden.ref;
  let history12: np.Array | null = null;
  let history23: np.Array | null = null;

  for (let i = 0; i < C.layers; i++) {
    const [attentionInput, residualBase] = layerInputs(
      model,
      hidden,
      i,
      history0,
      history12,
      history23,
    );
    state.caches[i]?.data.dispose();
    const previous = hidden;
    [hidden, state.caches[i]] = runLayerPrefill(
      tree.ref(model.layers[i]),
      attentionInput,
      residualBase,
      valueEmbeds.ref,
      ropeCosine.ref,
      ropeSine.ref,
      GLOBAL.has(i),
      VALUE.has(i),
      state.capacity,
    );
    // Submit each layer without reading back to the CPU. Otherwise JAX's lazy
    // graph retains unpacked Q4 weights from all 48 layers during prefill.
    // Submission releases scratch references so the next layer can reuse them.
    hidden.ref.gpuBufferSync();
    previous.dispose();
    if (i === 11) history12 = hidden.ref;
    if (i === 22) history23 = hidden.ref;
  }

  hidden = rmsNorm(hidden).slice(np.array([promptLength - 1], { dtype: np.int32 }));
  state.position = promptLength;
  valueEmbeds.dispose();
  ropeCosine.dispose();
  ropeSine.dispose();
  history0.dispose();
  history12?.dispose();
  history23?.dispose();
  return hidden;
}

export function step(model: LimiteModel, token: number | np.Array, state: LimiteState): np.Array {
  ensureCapacity(state, state.position + 1);
  const tokenIds = typeof token === "number"
    ? np.array([token], { dtype: np.uint32 }) : token.reshape([1]);
  const position = np.array(state.position, { dtype: np.uint32 });
  const [hidden, caches] = runStep(tree.ref(model), state.caches, tokenIds, position);
  state.caches = caches;
  state.position++;
  return hidden;
}

const runStep = jit(function runStep(
  model: LimiteModel,
  caches: KV[],
  tokenIds: np.Array,
  position: np.Array,
): [np.Array, KV[]] {
  let hidden = q4Embedding(tree.ref(model.embedTokens), tokenIds.ref);
  hidden = rmsNorm(hidden);
  const valueEmbeds = q4Embedding(tree.ref(model.valueEmbeds), tokenIds)
    .reshape([1, C.kvHeads, C.headDim]);
  const history0 = hidden.ref;
  let history12: np.Array | null = null;
  let history23: np.Array | null = null;
  const [ropeCosine, ropeSine] = makeRoPEFactors(position.ref, 1);
  const localSlot = np.remainder(position.ref, C.slidingWindow + 1);
  const validLength = position.ref.add(1);
  const localLength = np.minimum(validLength.ref, C.slidingWindow + 1);
  const nextCaches: KV[] = [];

  for (let i = 0; i < C.layers; i++) {
    const [attentionInput, residualBase] = layerInputs(
      model,
      hidden,
      i,
      history0,
      history12,
      history23,
    );
    const previous = hidden;
    [hidden, nextCaches[i]] = runLayerStep(
      tree.ref(model.layers[i]),
      caches[i],
      attentionInput,
      residualBase,
      valueEmbeds.ref,
      ropeCosine.ref,
      ropeSine.ref,
      GLOBAL.has(i) ? position.ref : localSlot.ref,
      GLOBAL.has(i) ? validLength.ref : localLength.ref,
      GLOBAL.has(i),
      VALUE.has(i),
    );
    previous.dispose();
    if (i === 11) history12 = hidden.ref;
    if (i === 22) history23 = hidden.ref;
  }

  hidden = rmsNorm(hidden);
  valueEmbeds.dispose();
  ropeCosine.dispose();
  ropeSine.dispose();
  history0.dispose();
  history12?.dispose();
  history23?.dispose();
  tree.dispose([model, position, localSlot, validLength, localLength]);
  return [hidden, nextCaches];
});

export async function loadQ4Model(onProgress?: LimiteArtifactProgressCallback): Promise<LimiteModel> {
  const matrices: Record<string, Q4Linear> = {};
  const small: Record<string, np.Array> = {};
  try {
    await readLimiteQ4Shards(async (manifest, index, bytes) => {
      for (const [name, tensor] of Object.entries(manifest.tensors)) {
        if (tensor.shard !== index) continue;
        if (tensor.dtype === "q4_block32") {
          matrices[name] = {
            codes: np.array(new Uint32Array(bytes.buffer, bytes.byteOffset + tensor.offset, tensor.byteLength / 4), {
              shape: [tensor.shape[0], tensor.shape[1] / 8],
            }),
            scales: np.array(new Float16Array(bytes.buffer, bytes.byteOffset + tensor.scaleOffset, tensor.scaleByteLength / 2), {
              shape: [...tensor.scaleShape], dtype: np.float16,
            }),
          };
          await blockUntilReady(matrices[name]);
        } else {
          small[name] = np.array(new Float32Array(bytes.buffer, bytes.byteOffset + tensor.offset, tensor.byteLength / 4), {
            shape: [...tensor.shape],
          });
          await blockUntilReady(small[name]);
        }
      }
    }, onProgress);
    const layers = Array.from({ length: C.layers }, (_, index): Layer => {
      const prefix = `layers.${index}`;
      const lambdas = small[`${prefix}.lambdas`];
      const layer = {
        selfAttn: {
          qkvProj: matrices[`${prefix}.qkv`], oProj: matrices[`${prefix}.o`],
          // Projection scales and tanh(alpha) are already folded into the Q4 artifact.
          xsaFactor: small[`${prefix}.xsa_alpha`],
        },
        mlp: { gateUpProj: matrices[`${prefix}.gate_up`], downProj: matrices[`${prefix}.down`] },
        residLambdaAttn: lambdas.ref.slice(0), postLambdaAttn: lambdas.ref.slice(1),
        residLambdaMlp: lambdas.ref.slice(2), postLambdaMlp: lambdas.slice(3),
      };
      delete small[`${prefix}.lambdas`];
      return layer;
    });
    return {
      embedTokens: matrices.embed_tokens, valueEmbeds: matrices.value_embeds, layers,
      mudd: { dense1: small["mudd.dense1"], dense2: small["mudd.dense2"],
        dense2Mlp: small["mudd.dense2_mlp"], bias: small["mudd.bias"], biasMlp: small["mudd.bias_mlp"] },
    };
  } catch (error) {
    tree.dispose(matrices); tree.dispose(small);
    throw error;
  }
}

export function disposeState(state: LimiteState): void {
  tree.dispose(state.caches);
  state.caches = [];
  state.position = 0;
}
