import { blockUntilReady, jit, nn, numpy as np, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export const MODEL_URL =
  "https://huggingface.co/gustofied/kalkulator/resolve/main/model-fp16.safetensors";
export const MODEL_BYTES = 2_070_811_520;
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
  qkvProj?: Linear;
  qProj?: Linear;
  kProj?: Linear;
  vProj?: Linear;
  oProj: Linear;
  qkvScale: np.Array;
  oScale: np.Array;
  xsaAlpha: np.Array;
  attnGate?: np.Array;
  veGate?: np.Array;
};
type MLP = {
  gateUpProj?: Linear;
  gateProj?: Linear;
  upProj?: Linear;
  downProj: Linear;
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
  embedTokens: Linear;
  valueEmbeds: Linear;
  layers: Layer[];
  mudd: Mudd;
};
type KV = { data: np.Array };
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

function projectAttention(
  attn: Attention,
  hidden: np.Array,
): {
  query: np.Array;
  key: np.Array;
  value: np.Array;
  attentionGate: np.Array;
  valueGate?: np.Array;
} {
  if (attn.qkvProj) {
    const projected = runLinear(attn.qkvProj, hidden);
    const qEnd = C.heads * C.headDim;
    const kEnd = qEnd + C.kvHeads * C.headDim;
    const vEnd = kEnd + C.kvHeads * C.headDim;
    const gateEnd = vEnd + C.heads;
    const qkv = projected.ref.slice([], [0, vEnd]).mul(attn.qkvScale);
    return {
      query: qkv.ref.slice([], [0, qEnd]),
      key: qkv.ref.slice([], [qEnd, kEnd]),
      value: qkv.slice([], [kEnd]),
      attentionGate: projected.ref.slice([], [vEnd, gateEnd]),
      valueGate:
        projected.shape[1] > gateEnd
          ? projected.slice([], [gateEnd])
          : undefined,
    };
  }
  const attentionChannels = np.take(
    hidden.ref,
    np.arange(128, undefined, undefined, { dtype: np.uint32 }),
    -1,
  );
  const valueChannels = attn.veGate
    ? np.take(
        hidden.ref,
        np.arange(12, undefined, undefined, { dtype: np.uint32 }),
        -1,
      )
    : undefined;
  return {
    query: project(attn.qProj!, attn.qkvScale.ref, hidden.ref),
    key: project(attn.kProj!, attn.qkvScale.ref, hidden.ref),
    value: project(attn.vProj!, attn.qkvScale, hidden.ref),
    attentionGate: np.dot(attentionChannels, attn.attnGate!.transpose()),
    valueGate: valueChannels
      ? np.dot(valueChannels, attn.veGate!.transpose())
      : undefined,
  };
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
    const cosine = np.cos(theta.ref).reshape([length, 1, C.headDim]);
    let sine = np.sin(theta);
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
  q: np.Array,
  k: np.Array,
  cosine: np.Array,
  sine: np.Array,
): [np.Array, np.Array] {
  const [T, qHeads, D] = q.shape;
  const kHeads = k.shape[1];

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
  const alpha = np
    .tanh(attn.xsaAlpha)
    .reshape([1, C.kvHeads, C.groups, 1]);
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
  const projected = projectAttention(attn, hidden.ref);
  let { query: q, key: k, value: v } = projected;
  q = q.reshape([
    T,
    C.heads,
    C.headDim,
  ]);
  k = k.reshape([
    T,
    C.kvHeads,
    C.headDim,
  ]);
  v = v.reshape([
    T,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) {
    v = applyValueEmbedding(projected.valueGate!, valueEmbeds, v);
  }
  const currentValues = v.ref;
  q = rmsNorm(q);
  k = rmsNorm(k);
  if (!isGlobal) [q, k] = applyRoPE(q, k, ropeCosine, ropeSine);

  let output = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: C.attentionScale,
    localWindowSize: isGlobal ? undefined : [C.slidingWindow, 0],
  });
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(projected.attentionGate, output);
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
  ropeCosine: np.Array,
  ropeSine: np.Array,
  position: number,
  slot: number,
  validLength: number,
  isGlobal: boolean,
  hasValueEmbedding: boolean,
): { output: np.Array; cache: KV } {
  const projected = projectAttention(attn, hidden.ref);
  let { query: q, key: k, value: v } = projected;
  q = q.reshape([
    1,
    C.heads,
    C.headDim,
  ]);
  k = k.reshape([
    1,
    C.kvHeads,
    C.headDim,
  ]);
  v = v.reshape([
    1,
    C.kvHeads,
    C.headDim,
  ]);
  if (hasValueEmbedding) {
    v = applyValueEmbedding(projected.valueGate!, valueEmbeds, v);
  }
  const currentValues = v.ref;
  q = rmsNorm(q);
  k = rmsNorm(k);
  if (!isGlobal) [q, k] = applyRoPE(q, k, ropeCosine, ropeSine);

  const capacity = cache.data.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1, 1]);
  const update = np.stack([k, v], 1);
  const data = np.where(
    slotMask,
    np.tile(update, [capacity, 1, 1, 1]),
    cache.data,
  );
  const key = data.ref.slice([], 0);
  const value = data.ref.slice([], 1);
  let mask = np.arange(capacity).less(validLength);
  if (!isGlobal) {
    mask = mask.mul(
      np.arange(capacity).greaterEqual(position - C.slidingWindow),
    );
  }
  let output = nn.dotProductAttention(q, key, value, {
    mask,
    scale: C.attentionScale,
  });
  output = applyXsa(attn, output, currentValues);
  output = applyAttentionGate(projected.attentionGate, output);
  output = project(
    attn.oProj,
    attn.oScale,
    output.reshape([1, C.heads * C.headDim]),
  );
  return { output, cache: { data } };
}

function runMlp(mlp: MLP, hidden: np.Array): np.Array {
  if (mlp.gateUpProj) {
    const gateUp = runLinear(mlp.gateUpProj, hidden);
    const gate = nn.silu(gateUp.ref.slice([], [0, C.intermediate]));
    const up = gateUp.slice([], [C.intermediate]);
    return runLinear(mlp.downProj, gate.mul(up));
  }
  const gate = nn.silu(runLinear(mlp.gateProj!, hidden.ref));
  const up = runLinear(mlp.upProj!, hidden);
  return runLinear(mlp.downProj, gate.mul(up));
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
    return [result, padCache(np.stack([key, value], 1), capacity)];
  },
  { staticArgnums: [6, 7, 8] },
);

const runLayerStep = jit(
  function runLayerStep(
    layer: Layer,
    cache: KV,
    attentionInput: np.Array,
    residualBase: np.Array,
    valueEmbeds: np.Array,
    ropeCosine: np.Array,
    ropeSine: np.Array,
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
      ropeCosine,
      ropeSine,
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
  { staticArgnums: [10, 11] },
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
    // Prefill creates every cache directly. Avoid allocating and immediately
    // disposing a capacity-sized zero cache for all 48 layers.
    caches: [],
  };
}

function ensureCapacity(state: LimiteState, required: number): void {
  if (state.capacity >= required) return;
  const old = state.capacity;
  const next = roundCapacity(required);
  for (const cache of state.caches) {
    cache.data = np.pad(cache.data, { 0: [0, next - old] });
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
  // Violetto's sigmoid softcap is strictly monotonic, so top-k can operate on
  // raw logits. The sampler applies the softcap to only the 50 survivors.
  return runLinear(model.embedTokens, hidden).astype(np.float32).reshape([C.vocab]);
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
  const [ropeCosine, ropeSine] = makeRoPEFactors(
    np.array(0, { dtype: np.uint32 }),
    tokenIds.shape[0],
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
      model.layers[i],
      attentionInput,
      residualBase,
      valueEmbeds.ref,
      ropeCosine.ref,
      ropeSine.ref,
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
  ropeCosine.dispose();
  ropeSine.dispose();
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
  const positionArray = np.array(position, { dtype: np.uint32 });
  const [ropeCosine, ropeSine] = makeRoPEFactors(positionArray.ref, 1);
  positionArray.dispose();

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
      ropeCosine.ref,
      ropeSine.ref,
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
  ropeCosine.dispose();
  ropeSine.dispose();
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

async function fuseProjectionWeights(model: LimiteModel): Promise<void> {
  // Violetto's native inference graph uses a single QKV projection and a
  // single gate/up projection. The browser artifact stores those matrices
  // separately, so fuse them once after upload to remove matvec dispatches per
  // layer and token. Gate weights are cast to the checkpoint dtype here, as
  // the upstream evaluation path does before applying them.
  const batchSize = 4;
  for (let start = 0; start < model.layers.length; start += batchSize) {
    const batch = model.layers.slice(start, start + batchSize).map((layer) => {
      const attn = layer.selfAttn;
      const mlp = layer.mlp;
      const attentionGate = np.pad(
        attn.attnGate!.ref.astype(np.float16),
        { 1: [0, C.hidden - 128] },
      );
      const valueGate = attn.veGate
        ? np.pad(attn.veGate.ref.astype(np.float16), {
            1: [0, C.hidden - 12],
          })
        : null;
      const qkv = np.concatenate(
        [
          attn.qProj!.weight.ref,
          attn.kProj!.weight.ref,
          attn.vProj!.weight.ref,
          attentionGate,
          ...(valueGate ? [valueGate] : []),
        ],
        0,
      );
      const gateUp = np.concatenate(
        [mlp.gateProj!.weight.ref, mlp.upProj!.weight.ref],
        0,
      );
      return { attn, mlp, qkv, gateUp };
    });

    await blockUntilReady(batch.map(({ qkv, gateUp }) => [qkv, gateUp]));
    for (const { attn, mlp, qkv, gateUp } of batch) {
      attn.qProj!.weight.dispose();
      attn.kProj!.weight.dispose();
      attn.vProj!.weight.dispose();
      attn.attnGate!.dispose();
      attn.veGate?.dispose();
      mlp.gateProj!.weight.dispose();
      mlp.upProj!.weight.dispose();
      attn.qkvProj = { weight: qkv };
      mlp.gateUpProj = { weight: gateUp };
      delete attn.qProj;
      delete attn.kProj;
      delete attn.vProj;
      delete attn.attnGate;
      delete attn.veGate;
      delete mlp.gateProj;
      delete mlp.upProj;
    }
  }
}

export async function loadModel(
  data: Uint8Array<ArrayBuffer> | ArrayBuffer,
): Promise<LimiteModel> {
  const file = safetensors.parse(data);
  const flat: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    flat[mapper.mapKey(key)] = tensorToArray(tensor);
  }
  const model = safetensors.toNested(flat) as LimiteModel;
  if (model.layers.length !== C.layers) throw new Error("Incomplete checkpoint");
  await blockUntilReady(model);
  await fuseProjectionWeights(model);
  return model;
}
