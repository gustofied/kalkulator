import { blockUntilReady, jit, nn, numpy as np, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export const MODEL_URL =
  "https://huggingface.co/gustofied/kalkulator/resolve/main/model-fp16.safetensors";
export const TOKENIZER_URL =
  "https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/main/tokenizer.json";

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

type Linear = { weight: np.Array };
type Attention = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;
  qkvScale: np.Array;
  oScale: np.Array;
  xsaAlpha: np.Array;
  attnGate: np.Array;
  veGate?: np.Array;
};
type MLP = { gateProj: Linear; upProj: Linear; downProj: Linear };
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
  embedTokens: Linear;
  valueEmbeds: Linear;
  layers: Layer[];
  mudd: Mudd;
};
type KV = { key: np.Array; value: np.Array };
export type LimiteState = { caches: KV[]; position: number; capacity: number };

const GLOBAL = new Set([3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]);
const VALUE = new Set([1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46]);

const runLinear = jit(function runLinear(linear: Linear, x: np.Array): np.Array {
  return np.dot(x, linear.weight.transpose());
});

const rmsNorm = jit(function rmsNorm(x: np.Array): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const meanSquare = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  return x.div(np.sqrt(meanSquare.add(C.rmsEps))).astype(dtype);
});

function project(linear: Linear, scale: np.Array, x: np.Array): np.Array {
  return runLinear(linear, x).mul(scale);
}

function applyRoPE(q: np.Array, k: np.Array, offset: number): [np.Array, np.Array] {
  const [T, qHeads, D] = q.shape;
  const kHeads = k.shape[1];
  const base = np.exp(np.linspace(0, 1, 32).mul(-Math.log(1024)));
  const frequency = np.concatenate([
    np.repeat(base, 2),
    np.zeros([D - 64], { dtype: np.float32 }),
  ]);
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const theta = positions.mul(frequency.reshape([1, D]));
  const cosine = np.cos(theta.ref).reshape([T, 1, D]);
  let sine = np.sin(theta);
  const odd = np.remainder(np.arange(D), 2).equal(1).reshape([1, D]);
  sine = np.where(odd, sine.ref.mul(-1), sine).reshape([T, 1, D]);

  const qPaired = np.flip(q.ref.reshape([T, qHeads, D / 2, 2]), -1).reshape([
    T,
    qHeads,
    D,
  ]);
  const kPaired = np.flip(k.ref.reshape([T, kHeads, D / 2, 2]), -1).reshape([
    T,
    kHeads,
    D,
  ]);
  return [
    q.mul(cosine.ref).add(qPaired.mul(sine.ref)),
    k.mul(cosine).add(kPaired.mul(sine)),
  ];
}

function applyValueEmbedding(
  attn: Attention,
  hidden: np.Array,
  valueEmbeds: np.Array,
  values: np.Array,
): np.Array {
  const channels = np.take(
    hidden,
    np.arange(12, undefined, undefined, { dtype: np.uint32 }),
    -1,
  );
  const gate = nn.sigmoid(np.dot(channels, attn.veGate!.transpose())).mul(2);
  return values.add(gate.reshape([hidden.shape[0], C.kvHeads, 1]).mul(valueEmbeds));
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
  const alpha = np
    .tanh(attn.xsaAlpha)
    .reshape([1, C.kvHeads, C.groups, 1]);
  return grouped
    .sub(alpha.mul(projection).mul(direction))
    .reshape([T, C.heads, C.headDim]);
}

function applyAttentionGate(attn: Attention, hidden: np.Array, output: np.Array): np.Array {
  const channels = np.take(
    hidden,
    np.arange(128, undefined, undefined, { dtype: np.uint32 }),
    -1,
  );
  const gate = nn.sigmoid(np.dot(channels, attn.attnGate.transpose())).mul(2);
  return output.mul(gate.reshape([hidden.shape[0], C.heads, 1]));
}

function attentionPrefill(
  attn: Attention,
  hidden: np.Array,
  valueEmbeds: np.Array,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = hidden.shape[0];
  let q = project(attn.qProj, attn.qkvScale.ref, hidden.ref).reshape([
    T,
    C.heads,
    C.headDim,
  ]);
  let k = project(attn.kProj, attn.qkvScale.ref, hidden.ref).reshape([
    T,
    C.kvHeads,
    C.headDim,
  ]);
  let v = project(attn.vProj, attn.qkvScale, hidden.ref).reshape([
    T,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) v = applyValueEmbedding(attn, hidden.ref, valueEmbeds, v);
  const currentValues = v.ref;
  q = rmsNorm(q);
  k = rmsNorm(k);
  if (!isGlobal) [q, k] = applyRoPE(q, k, 0);

  let output = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: C.attentionScale,
    localWindowSize: isGlobal ? undefined : [C.slidingWindow, 0],
  });
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(attn, hidden, output);
  output = project(
    attn.oProj,
    attn.oScale,
    output.reshape([T, C.heads * C.headDim]),
  );
  return { output, key: k, value: v };
}

function attentionStep(
  attn: Attention,
  cache: KV,
  hidden: np.Array,
  valueEmbeds: np.Array,
  position: number,
  slot: number,
  validLength: number,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): { output: np.Array; cache: KV } {
  let q = project(attn.qProj, attn.qkvScale.ref, hidden.ref).reshape([
    1,
    C.heads,
    C.headDim,
  ]);
  let k = project(attn.kProj, attn.qkvScale.ref, hidden.ref).reshape([
    1,
    C.kvHeads,
    C.headDim,
  ]);
  let v = project(attn.vProj, attn.qkvScale, hidden.ref).reshape([
    1,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) v = applyValueEmbedding(attn, hidden.ref, valueEmbeds, v);
  const currentValues = v.ref;
  q = rmsNorm(q);
  k = rmsNorm(k);
  if (!isGlobal) [q, k] = applyRoPE(q, k, position);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);
  let mask = np.arange(capacity).less(validLength);
  if (!isGlobal) {
    mask = mask.mul(
      np.arange(capacity).greaterEqual(position - C.slidingWindow),
    );
  }

  let output = nn.dotProductAttention(q, key.ref, value.ref, {
    mask,
    scale: C.attentionScale,
  });
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(attn, hidden, output);
  output = project(
    attn.oProj,
    attn.oScale,
    output.reshape([1, C.heads * C.headDim]),
  );
  return { output, cache: { key, value } };
}

function runMlp(mlp: MLP, hidden: np.Array): np.Array {
  const gate = nn.silu(runLinear(mlp.gateProj, hidden.ref));
  const up = runLinear(mlp.upProj, hidden);
  return runLinear(mlp.downProj, gate.mul(up));
}

function padCache(key: np.Array, value: np.Array, capacity: number): KV {
  const T = key.shape[0];
  if (T > capacity) throw new Error("Prompt exceeds context capacity");
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

const runLayerPrefill = jit(
  function runLayerPrefill(
    layer: Layer,
    attentionInput: np.Array,
    residualBase: np.Array,
    valueEmbeds: np.Array,
    isGlobal: boolean,
    hasValueEmbedding: boolean,
    capacity: number,
  ): [np.Array, KV] {
    const { output, key, value } = attentionPrefill(
      layer.selfAttn,
      attentionInput,
      valueEmbeds,
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
    return [result, padCache(key, value, capacity)];
  },
  { staticArgnums: [4, 5, 6] },
);

const runLayerStep = jit(
  function runLayerStep(
    layer: Layer,
    cache: KV,
    attentionInput: np.Array,
    residualBase: np.Array,
    valueEmbeds: np.Array,
    position: number,
    slot: number,
    validLength: number,
    isGlobal: boolean,
    hasValueEmbedding: boolean,
  ): [np.Array, KV] {
    const { output, cache: nextCache } = attentionStep(
      layer.selfAttn,
      cache,
      attentionInput,
      valueEmbeds,
      position,
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
  },
  { staticArgnums: [8, 9] },
);

const runMuddPair = jit(
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
  },
  { staticArgnums: [5] },
);

function roundCapacity(required: number): number {
  return Math.max(C.cacheBlock, Math.ceil(required / C.cacheBlock) * C.cacheBlock);
}

export function createState(capacity: number = C.cacheBlock): LimiteState {
  capacity = roundCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: C.layers }, () => ({
      key: np.zeros([capacity, C.kvHeads, C.headDim], { dtype: np.float32 }),
      value: np.zeros([capacity, C.kvHeads, C.headDim], { dtype: np.float32 }),
    })),
  };
}

function ensureCapacity(state: LimiteState, required: number): void {
  if (state.capacity >= required) return;
  const old = state.capacity;
  const next = roundCapacity(required);
  for (const cache of state.caches) {
    cache.key = np.pad(cache.key, { 0: [0, next - old] });
    cache.value = np.pad(cache.value, { 0: [0, next - old] });
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
      model.mudd,
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

function finishLogits(model: LimiteModel, hidden: np.Array): np.Array {
  const logits = runLinear(model.embedTokens, hidden).astype(np.float32);
  return nn.sigmoid(logits.add(5).div(7.5)).mul(23).reshape([C.vocab]);
}

export function prefill(
  model: LimiteModel,
  tokenIds: np.Array,
  state: LimiteState,
): np.Array {
  ensureCapacity(state, tokenIds.shape[0]);
  let hidden = model.embedTokens.weight.ref.slice(tokenIds.ref).astype(np.float32);
  hidden = rmsNorm(hidden);
  const valueEmbeds = model.valueEmbeds.weight
    .slice(tokenIds)
    .astype(np.float32)
    .reshape([hidden.shape[0], C.kvHeads, C.headDim]);
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
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    const previous = hidden;
    [hidden, state.caches[i]] = runLayerPrefill(
      model.layers[i],
      attentionInput,
      residualBase,
      valueEmbeds.ref,
      GLOBAL.has(i),
      VALUE.has(i),
      state.capacity,
    );
    previous.dispose();
    if (i === 11) history12 = hidden.ref;
    if (i === 22) history23 = hidden.ref;
  }

  hidden = rmsNorm(hidden).slice([-1]);
  state.position = tokenIds.shape[0];
  valueEmbeds.dispose();
  history0.dispose();
  history12?.dispose();
  history23?.dispose();
  return finishLogits(model, hidden);
}

export function step(model: LimiteModel, token: number, state: LimiteState): np.Array {
  ensureCapacity(state, state.position + 1);
  const tokenIds = np.array([token], { dtype: np.uint32 });
  let hidden = model.embedTokens.weight.ref.slice(tokenIds.ref).astype(np.float32);
  hidden = rmsNorm(hidden);
  const valueEmbeds = model.valueEmbeds.weight
    .slice(tokenIds)
    .astype(np.float32)
    .reshape([1, C.kvHeads, C.headDim]);
  const history0 = hidden.ref;
  let history12: np.Array | null = null;
  let history23: np.Array | null = null;
  const position = state.position;

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
    [hidden, state.caches[i]] = runLayerStep(
      model.layers[i],
      state.caches[i],
      attentionInput,
      residualBase,
      valueEmbeds.ref,
      position,
      position,
      position + 1,
      GLOBAL.has(i),
      VALUE.has(i),
    );
    previous.dispose();
    if (i === 11) history12 = hidden.ref;
    if (i === 22) history23 = hidden.ref;
  }

  hidden = rmsNorm(hidden);
  state.position++;
  valueEmbeds.dispose();
  history0.dispose();
  history12?.dispose();
  history23?.dispose();
  return finishLogits(model, hidden);
}

const mapper = new WeightMapper({
  prefix: { "model.": "" },
  substring: {
    embed_tokens: "embedTokens",
    value_embeds: "valueEmbeds",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    o_proj: "oProj",
    qkv_scale: "qkvScale",
    o_scale: "oScale",
    xsa_alpha: "xsaAlpha",
    attn_gate: "attnGate",
    ve_gate: "veGate",
    gate_proj: "gateProj",
    up_proj: "upProj",
    down_proj: "downProj",
    resid_lambda_attn: "residLambdaAttn",
    post_lambda_attn: "postLambdaAttn",
    resid_lambda_mlp: "residLambdaMlp",
    post_lambda_mlp: "postLambdaMlp",
    dense2_mlp: "dense2Mlp",
    bias_mlp: "biasMlp",
  },
});

function tensorToArray(tensor: safetensors.Tensor): np.Array {
  if (tensor.dtype === "F16") {
    return np.array(tensor.data as Float16Array<ArrayBuffer>, {
      shape: tensor.shape,
      dtype: np.float16,
    });
  }
  if (tensor.dtype === "F32") {
    return np.array(tensor.data as Float32Array<ArrayBuffer>, {
      shape: tensor.shape,
      dtype: np.float32,
    });
  }
  throw new Error(`Unsupported checkpoint dtype ${tensor.dtype}`);
}

export async function loadModel(data: ArrayBuffer): Promise<LimiteModel> {
  const file = safetensors.parse(data);
  const flat: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    flat[mapper.mapKey(key)] = tensorToArray(tensor);
  }
  const model = safetensors.toNested(flat) as LimiteModel;
  if (model.layers.length !== C.layers) throw new Error("Incomplete checkpoint");
  return blockUntilReady(model);
}
