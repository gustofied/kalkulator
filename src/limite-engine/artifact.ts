export const PINNED_FULL_BF16_MANIFEST_URL =
  "https://huggingface.co/gustofied/kalkulator/resolve/df708b2170cecfde605a77dd57cd5bb78f36c852/webgpu-full-bf16-v16/manifest.json" as const;

const LIMITE_MANIFEST_URL = resolveManifestUrl();
const USING_CUSTOM_DEV_MANIFEST =
  import.meta.env.DEV && LIMITE_MANIFEST_URL !== PINNED_FULL_BF16_MANIFEST_URL;
const ARTIFACT_FETCH_CACHE: RequestCache =
  USING_CUSTOM_DEV_MANIFEST ? "no-store" : "force-cache";
const KALKULATOR_ARTIFACT_CACHE_NAME =
  /^kalkulator-(?:q4|model-v\d+)-[0-9a-f]{20,64}\.bin$/;

const MANIFEST_VERSION = 16;
const EXPECTED_SHARD_BYTE_LENGTHS = [
  115_200_000,
  115_200_000,
  115_200_000,
  42_700_800,
  115_727_872,
  103_630_336,
  117_392_896,
  116_763_648,
  117_392_896,
  116_763_648,
  117_392_896,
  116_768_768,
  117_392_896,
  116_763_648,
  117_392_896,
  116_763_648,
  117_392_896,
  116_768_768,
  60_259_840,
] as const;
const SHARD_COUNT = EXPECTED_SHARD_BYTE_LENGTHS.length;
const ALIGNMENT = 256;
const SHARD_MAX_BYTES = 112 * 1024 * 1024;
const EMBED_SEGMENT_ROWS = 45_000;
const LAYER_COUNT = 48;
const HIDDEN_SIZE = 1_280;
const INTERMEDIATE_SIZE = 3_328;
const QUERY_HEADS = 10;
const KV_HEADS = 2;
const HEAD_DIM = 128;
const VOCAB_SIZE = 151_680;
const SHA256_PATTERN = /^[0-9a-f]{64}$/;

const GLOBAL_LAYERS = Array.from({ length: 12 }, (_, index) => index * 4 + 3);
const VALUE_LAYERS = Array.from({ length: 16 }, (_, index) => index * 3 + 1);
const XSA_LAYERS = Array.from({ length: LAYER_COUNT }, (_, index) => index);
const VALUE_LAYER_SET = new Set(VALUE_LAYERS);

function resolveManifestUrl(): string {
  if (!import.meta.env.DEV) return PINNED_FULL_BF16_MANIFEST_URL;

  const configured = import.meta.env.VITE_LIMITE_MANIFEST_URL?.trim();
  if (!configured) return PINNED_FULL_BF16_MANIFEST_URL;

  const url = new URL(configured);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("VITE_LIMITE_MANIFEST_URL must use HTTP or HTTPS.");
  }
  return url.href;
}

export type LimiteFullBf16SourceManifest = {
  readonly repo: "paradigma-inc/limite-1b-violetto";
  readonly revision: "b1f3d572ccacb6919f4d64c321b70ba034ddaef2";
  readonly file: "model.safetensors";
  readonly sha256: "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5";
  readonly byteLength: 2_070_811_520;
};

export type LimiteFullBf16FormatManifest = {
  readonly alignment: 256;
  readonly shardMaxBytes: 117_440_512;
  readonly tiedEmbedding: {
    readonly dtype: "bf16";
    readonly packing: "two_bf16_values_per_u32_little_endian";
    readonly matrixLayout: "output_major";
    readonly roles: readonly ["token_embedding", "lm_head"];
  };
  readonly valueEmbedding: {
    readonly dtype: "bf16";
    readonly packing: "two_bf16_values_per_u32_little_endian";
    readonly matrixLayout: "output_major";
    readonly precision: "oracle_exact";
    readonly roles: readonly ["value_embedding"];
  };
  readonly bodyPrecision: {
    readonly default: "bf16";
    readonly exactBf16Layers: readonly number[];
    readonly exactBf16Projections: readonly ["qkv", "o", "gate_up", "down"];
    readonly packing: "two_bf16_values_per_u32_little_endian";
    readonly matrixLayout: "output_major";
  };
};

export type LimiteModelManifest = {
  readonly model_type: "limite";
  readonly hidden_size: 1_280;
  readonly intermediate_size: 3_328;
  readonly num_hidden_layers: 48;
  readonly num_attention_heads: 10;
  readonly num_key_value_heads: 2;
  readonly head_dim: 128;
  readonly vocab_size: 151_680;
  readonly padded_vocab_size: 151_680;
  readonly tokenizer_vocab_size: 151_667;
  readonly bos_token_id: 151_643;
  readonly eos_token_id: 151_643;
  readonly pad_token_id: 151_643;
  readonly max_position_embeddings: 131_072;
  readonly sliding_window: 1_024;
  readonly attention_softmax_scale: 0.1;
  readonly global_layers: readonly number[];
  readonly global_nope: true;
  readonly rms_norm_has_weight: false;
  readonly rms_norm_eps_mode: "torch_finfo_default";
  readonly qk_norm: "rms_pre_rope";
  readonly rope_base_global: 1_024;
  readonly rope_base_local: 1_024;
  readonly rope_frac: 0.5;
  readonly rope_n_pairs: 32;
  readonly rope_style: "interleaved_pairs_odd_lane_sign_flip";
  readonly tie_word_embeddings: true;
  readonly lm_head_precision_mode: "oracle_exact";
  readonly value_embedding_precision_mode: "oracle_exact";
  readonly body_precision_mode: "full_bf16_oracle_exact";
  readonly ve_layers: readonly number[];
  readonly ve_dim: 128;
  readonly ve_stored_heads: 2;
  readonly ve_gate_channels: 12;
  readonly ve_gate_scale: 2;
  readonly xsa: true;
  readonly xsa_layers: readonly number[];
  readonly xsa_normalize_eps: 0.0001;
  readonly mudd: true;
  readonly mudd_layers: readonly [24, 47];
  readonly mudd_at: readonly [24, 47];
  readonly mudd_inter: 32;
  readonly mudd_taps: 3;
  readonly mudd_tap_idx: {
    readonly "24": readonly [0, 12, 24];
    readonly "47": readonly [0, 23, 47];
  };
  readonly softcap_logits: {
    readonly kind: "sigmoid";
    readonly a: 23;
    readonly b: 5;
    readonly c: 7.5;
  };
};

export type LimiteShardManifest = {
  readonly file: string;
  readonly url: string;
  readonly byteLength: number;
  readonly sha256: string;
};

export type LimiteTensorPart = {
  readonly name: string;
  readonly rowStart: number;
  readonly rowCount: number;
};

type LimiteTensorBase = {
  readonly shard: number;
  readonly offset: number;
  readonly offsetWords: number;
  readonly byteLength: number;
  readonly shape: readonly number[];
};

type LimitePackedBf16TensorBase = LimiteTensorBase & {
  readonly dtype: "bf16";
  readonly shape: readonly [number, number];
};

export type LimiteTiedPackedBf16TensorManifest = LimitePackedBf16TensorBase & {
  readonly roles: readonly ["token_embedding", "lm_head"];
  readonly rowStart: number;
};

export type LimiteValuePackedBf16TensorManifest = LimitePackedBf16TensorBase & {
  readonly roles: readonly ["value_embedding"];
  readonly rowStart: 0;
  readonly rowShape: readonly [2, 128];
};

export type LimiteEmbeddingPackedBf16TensorManifest =
  | LimiteTiedPackedBf16TensorManifest
  | LimiteValuePackedBf16TensorManifest;

export type LimiteBodyPackedBf16TensorManifest = LimitePackedBf16TensorBase & {
  readonly roles: readonly [string];
  readonly precision: "oracle_exact";
  readonly parts?: readonly LimiteTensorPart[];
};

export type LimitePackedBf16TensorManifest =
  | LimiteEmbeddingPackedBf16TensorManifest
  | LimiteBodyPackedBf16TensorManifest;

export type LimiteSmallTensorManifest = LimiteTensorBase & {
  readonly dtype: "f32";
  readonly transform?: "tanh";
  readonly parts?: readonly string[];
};

export type LimiteTensorManifest =
  | LimitePackedBf16TensorManifest
  | LimiteSmallTensorManifest;

export type LimiteFullBf16Manifest = {
  readonly version: 16;
  readonly source: LimiteFullBf16SourceManifest;
  readonly format: LimiteFullBf16FormatManifest;
  readonly model: LimiteModelManifest;
  readonly shards: readonly LimiteShardManifest[];
  readonly tensors: Readonly<Record<string, LimiteTensorManifest>>;
};

export type LimiteArtifactProgress =
  | { readonly phase: "manifest" }
  | {
      readonly phase: "shards";
      readonly source: "cache" | "network";
      readonly loadedBytes: number;
      readonly totalBytes: number;
      readonly shardIndex: number;
      readonly shardLoadedBytes: number;
      readonly shardByteLength: number;
    };

export type LimiteArtifactProgressCallback = (progress: LimiteArtifactProgress) => void;

type ArtifactCacheSession = {
  readonly root: FileSystemDirectoryHandle;
  writeEnabled: boolean;
};

const XSA_OFFSET = 0;
const XSA_LENGTH = LAYER_COUNT * QUERY_HEADS;
const LAMBDAS_OFFSET = XSA_OFFSET + XSA_LENGTH;
const LAMBDAS_LENGTH = LAYER_COUNT * 4;
const MUDD_DENSE1_OFFSET = LAMBDAS_OFFSET + LAMBDAS_LENGTH;
const MUDD_DENSE1_LENGTH = 32 * HIDDEN_SIZE;
const MUDD_DENSE2_OFFSET = MUDD_DENSE1_OFFSET + MUDD_DENSE1_LENGTH;
const MUDD_DENSE2_LENGTH = LAYER_COUNT * 3 * 32;
const MUDD_DENSE2_MLP_OFFSET = MUDD_DENSE2_OFFSET + MUDD_DENSE2_LENGTH;
const MUDD_DENSE2_MLP_LENGTH = LAYER_COUNT * 3 * 32;
const MUDD_BIAS_OFFSET = MUDD_DENSE2_MLP_OFFSET + MUDD_DENSE2_MLP_LENGTH;
const MUDD_BIAS_LENGTH = LAYER_COUNT * 3;
const MUDD_BIAS_MLP_OFFSET = MUDD_BIAS_OFFSET + MUDD_BIAS_LENGTH;
const MUDD_BIAS_MLP_LENGTH = LAYER_COUNT * 3;
const GELU_LOOKUP_OFFSET = MUDD_BIAS_MLP_OFFSET + MUDD_BIAS_MLP_LENGTH;
const BF16_LOOKUP_LENGTH = 1 << 16;
const SILU_LOOKUP_OFFSET = GELU_LOOKUP_OFFSET + BF16_LOOKUP_LENGTH;
const SIGMOID_LOOKUP_OFFSET = SILU_LOOKUP_OFFSET + BF16_LOOKUP_LENGTH;
const SAMPLING_WEIGHT_LOOKUP_OFFSET = SIGMOID_LOOKUP_OFFSET + BF16_LOOKUP_LENGTH;
const SMALL_TENSOR_ELEMENTS = SAMPLING_WEIGHT_LOOKUP_OFFSET + BF16_LOOKUP_LENGTH;

export const SMALL_TENSOR_LAYOUT = {
  xsa: XSA_OFFSET,
  lambdas: LAMBDAS_OFFSET,
  dense1: MUDD_DENSE1_OFFSET,
  dense2: MUDD_DENSE2_OFFSET,
  dense2Mlp: MUDD_DENSE2_MLP_OFFSET,
  bias: MUDD_BIAS_OFFSET,
  biasMlp: MUDD_BIAS_MLP_OFFSET,
  geluByBf16: GELU_LOOKUP_OFFSET,
  siluByBf16: SILU_LOOKUP_OFFSET,
  sigmoidByBf16: SIGMOID_LOOKUP_OFFSET,
  samplingWeightByBf16: SAMPLING_WEIGHT_LOOKUP_OFFSET,
} as const;

/** Shapes and row-major strides for the f32 arena addressed by SMALL_TENSOR_LAYOUT. */
export const SMALL_TENSOR_METADATA = {
  elementLength: SMALL_TENSOR_ELEMENTS,
  byteLength: SMALL_TENSOR_ELEMENTS * Float32Array.BYTES_PER_ELEMENT,
  xsa: { shape: [LAYER_COUNT, QUERY_HEADS], strides: [QUERY_HEADS, 1], length: XSA_LENGTH },
  lambdas: { shape: [LAYER_COUNT, 4], strides: [4, 1], length: LAMBDAS_LENGTH },
  dense1: { shape: [32, HIDDEN_SIZE], strides: [HIDDEN_SIZE, 1], length: MUDD_DENSE1_LENGTH },
  dense2: { shape: [LAYER_COUNT, 3, 32], strides: [96, 32, 1], length: MUDD_DENSE2_LENGTH },
  dense2Mlp: {
    shape: [LAYER_COUNT, 3, 32],
    strides: [96, 32, 1],
    length: MUDD_DENSE2_MLP_LENGTH,
  },
  bias: { shape: [LAYER_COUNT, 3], strides: [3, 1], length: MUDD_BIAS_LENGTH },
  biasMlp: { shape: [LAYER_COUNT, 3], strides: [3, 1], length: MUDD_BIAS_MLP_LENGTH },
  geluByBf16: { shape: [BF16_LOOKUP_LENGTH], strides: [1], length: BF16_LOOKUP_LENGTH },
  siluByBf16: { shape: [BF16_LOOKUP_LENGTH], strides: [1], length: BF16_LOOKUP_LENGTH },
  sigmoidByBf16: {
    shape: [BF16_LOOKUP_LENGTH],
    strides: [1],
    length: BF16_LOOKUP_LENGTH,
  },
  samplingWeightByBf16: {
    shape: [BF16_LOOKUP_LENGTH],
    strides: [1],
    length: BF16_LOOKUP_LENGTH,
  },
} as const;

export type LoadedLimiteFullBf16Artifact = {
  readonly manifest: LimiteFullBf16Manifest;
  readonly shards: readonly GPUBuffer[];
  readonly smallWeights: GPUBuffer;
  readonly smallValues: Float32Array;
  readonly smallLayout: typeof SMALL_TENSOR_LAYOUT;
  destroy(): void;
};

type Range = { readonly start: number; readonly end: number; readonly label: string };
type ExpectedPackedBf16 = {
  readonly shape: readonly [number, number];
  readonly extras: Readonly<Record<string, unknown>>;
};
type ExpectedSmall = {
  readonly shape: readonly number[];
  readonly extras: Readonly<Record<string, unknown>>;
};

const EXPECTED_PACKED_BF16 = new Map<string, ExpectedPackedBf16>();
const EXPECTED_SMALL = new Map<string, ExpectedSmall>();

for (let rowStart = 0, segment = 0; rowStart < VOCAB_SIZE; rowStart += EMBED_SEGMENT_ROWS) {
  EXPECTED_PACKED_BF16.set(`embed_tokens.${segment}`, {
    shape: [Math.min(EMBED_SEGMENT_ROWS, VOCAB_SIZE - rowStart), HIDDEN_SIZE],
    extras: { roles: ["token_embedding", "lm_head"], rowStart },
  });
  segment++;
}
EXPECTED_PACKED_BF16.set("value_embeds", {
  shape: [VOCAB_SIZE, KV_HEADS * HEAD_DIM],
  extras: {
    roles: ["value_embedding"],
    rowStart: 0,
    rowShape: [KV_HEADS, HEAD_DIM],
  },
});

for (let layer = 0; layer < LAYER_COUNT; layer++) {
  const prefix = `layers.${layer}`;
  const qkvParts: LimiteTensorPart[] = [
    { name: "q", rowStart: 0, rowCount: QUERY_HEADS * HEAD_DIM },
    {
      name: "k",
      rowStart: QUERY_HEADS * HEAD_DIM,
      rowCount: KV_HEADS * HEAD_DIM,
    },
    {
      name: "v",
      rowStart: (QUERY_HEADS + KV_HEADS) * HEAD_DIM,
      rowCount: KV_HEADS * HEAD_DIM,
    },
    {
      name: "attention_gate",
      rowStart: (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM,
      rowCount: QUERY_HEADS,
    },
  ];
  if (VALUE_LAYER_SET.has(layer)) {
    qkvParts.push({
      name: "value_gate",
      rowStart: (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM + QUERY_HEADS,
      rowCount: KV_HEADS,
    });
  }
  const qkvRows = qkvParts.reduce((count, part) => count + part.rowCount, 0);
  const bodyMatrices: readonly {
    readonly projection: "qkv" | "o" | "gate_up" | "down";
    readonly name: string;
    readonly shape: readonly [number, number];
    readonly extras: Readonly<Record<string, unknown>>;
  }[] = [
    {
      projection: "qkv",
      name: `${prefix}.qkv`,
      shape: [qkvRows, HIDDEN_SIZE],
      extras: { parts: qkvParts, roles: ["attention_qkv_and_gates_projection"] },
    },
    {
      projection: "o",
      name: `${prefix}.o`,
      shape: [HIDDEN_SIZE, HIDDEN_SIZE],
      extras: { roles: ["attention_output_projection"] },
    },
    {
      projection: "gate_up",
      name: `${prefix}.gate_up`,
      shape: [INTERMEDIATE_SIZE * 2, HIDDEN_SIZE],
      extras: {
        parts: [
          { name: "gate", rowStart: 0, rowCount: INTERMEDIATE_SIZE },
          {
            name: "up",
            rowStart: INTERMEDIATE_SIZE,
            rowCount: INTERMEDIATE_SIZE,
          },
        ],
        roles: ["mlp_gate_and_up_projection"],
      },
    },
    {
      projection: "down",
      name: `${prefix}.down`,
      shape: [HIDDEN_SIZE, INTERMEDIATE_SIZE],
      extras: { roles: ["mlp_down_projection"] },
    },
  ];
  for (const matrix of bodyMatrices) {
    EXPECTED_PACKED_BF16.set(matrix.name, {
      shape: matrix.shape,
      extras: { ...matrix.extras, precision: "oracle_exact" },
    });
  }
  EXPECTED_SMALL.set(`${prefix}.xsa_alpha`, {
    shape: [QUERY_HEADS],
    extras: { transform: "tanh" },
  });
  EXPECTED_SMALL.set(`${prefix}.lambdas`, {
    shape: [4],
    extras: {
      parts: [
        "resid_lambda_attn",
        "post_lambda_attn",
        "resid_lambda_mlp",
        "post_lambda_mlp",
      ],
    },
  });
}

EXPECTED_SMALL.set("mudd.dense1", { shape: [32, HIDDEN_SIZE], extras: {} });
EXPECTED_SMALL.set("mudd.dense2", { shape: [LAYER_COUNT, 3, 32], extras: {} });
EXPECTED_SMALL.set("mudd.dense2_mlp", {
  shape: [LAYER_COUNT, 3, 32],
  extras: {},
});
EXPECTED_SMALL.set("mudd.bias", { shape: [LAYER_COUNT, 3], extras: {} });
EXPECTED_SMALL.set("mudd.bias_mlp", { shape: [LAYER_COUNT, 3], extras: {} });
EXPECTED_SMALL.set("runtime.gelu_bf16", { shape: [BF16_LOOKUP_LENGTH], extras: {} });
EXPECTED_SMALL.set("runtime.silu_bf16", { shape: [BF16_LOOKUP_LENGTH], extras: {} });
EXPECTED_SMALL.set("runtime.sigmoid_bf16", { shape: [BF16_LOOKUP_LENGTH], extras: {} });
EXPECTED_SMALL.set("runtime.sampling_weight_by_bf16", {
  shape: [BF16_LOOKUP_LENGTH],
  extras: {},
});

/** Validates the fixed v16 full-BF16 artifact contract. */
export function validateLimiteFullBf16Manifest(value: unknown): LimiteFullBf16Manifest {
  const root = objectAt(value, "manifest");
  exactKeys(root, ["version", "source", "format", "model", "shards", "tensors"], "manifest");
  literal(root.version, MANIFEST_VERSION, "manifest.version");

  validateSource(root.source);
  validateFormat(root.format);
  validateModel(root.model);

  const shards = arrayAt(root.shards, "manifest.shards");
  if (shards.length !== SHARD_COUNT) {
    fail(`manifest.shards must contain exactly ${SHARD_COUNT} shards`);
  }
  const shardManifests = shards.map((entry, index) => validateShard(entry, index));

  const tensors = objectAt(root.tensors, "manifest.tensors");
  const expectedTensorCount = EXPECTED_PACKED_BF16.size + EXPECTED_SMALL.size;
  if (Object.keys(tensors).length !== expectedTensorCount) {
    fail(
      `manifest.tensors must contain exactly ${expectedTensorCount} tensors; found ${Object.keys(tensors).length}`,
    );
  }

  const rangesByShard = Array.from({ length: SHARD_COUNT }, () => [] as Range[]);
  for (const [name, expected] of EXPECTED_PACKED_BF16) {
    const tensor = tensors[name];
    if (tensor === undefined) fail(`manifest.tensors is missing ${name}`);
    validatePackedBf16Tensor(name, tensor, expected, shardManifests, rangesByShard);
  }
  for (const [name, expected] of EXPECTED_SMALL) {
    const tensor = tensors[name];
    if (tensor === undefined) fail(`manifest.tensors is missing ${name}`);
    validateSmallTensor(name, tensor, expected, shardManifests, rangesByShard);
  }
  for (const name of Object.keys(tensors)) {
    if (
      !EXPECTED_PACKED_BF16.has(name) &&
      !EXPECTED_SMALL.has(name)
    ) {
      fail(`manifest.tensors contains unexpected tensor ${name}`);
    }
  }

  for (let shard = 0; shard < rangesByShard.length; shard++) {
    const ranges = rangesByShard[shard].sort((left, right) => left.start - right.start);
    for (let index = 1; index < ranges.length; index++) {
      if (ranges[index].start < ranges[index - 1].end) {
        fail(
          `manifest shard ${shard} ranges overlap: ${ranges[index - 1].label} and ${ranges[index].label}`,
        );
      }
    }
  }

  return deepFreeze(root) as unknown as LimiteFullBf16Manifest;
}

export async function loadLimiteFullBf16Artifact(
  device: GPUDevice,
  onProgress: LimiteArtifactProgressCallback = () => {},
): Promise<LoadedLimiteFullBf16Artifact> {
  onProgress({ phase: "manifest" });
  const manifestResponse = await fetch(LIMITE_MANIFEST_URL, { cache: ARTIFACT_FETCH_CACHE });
  if (!manifestResponse.ok) {
    throw new Error(
      `Could not fetch Limite full-BF16 manifest (${manifestResponse.status} ${manifestResponse.statusText}).`,
    );
  }
  const manifest = validateLimiteFullBf16Manifest(await manifestResponse.json());
  const totalBytes = manifest.shards.reduce((total, shard) => total + shard.byteLength, 0);
  const cache = await prepareArtifactCache(manifest.shards, totalBytes);
  const smallValues = new Float32Array(SMALL_TENSOR_METADATA.elementLength);
  const shards: GPUBuffer[] = [];
  let loadedBytes = 0;

  try {
    for (let shardIndex = 0; shardIndex < manifest.shards.length; shardIndex++) {
      const shard = manifest.shards[shardIndex];
      const url = new URL(shard.url, LIMITE_MANIFEST_URL).href;
      const bytes = await fetchShard(
        url,
        shard,
        shardIndex,
        totalBytes,
        loadedBytes,
        onProgress,
        cache,
      );
      extractSmallTensors(manifest, shardIndex, bytes, smallValues);
      const buffer = immutableStorageBuffer(device, bytes, `Limite full BF16 ${shard.file}`);
      shards.push(buffer);
      loadedBytes += bytes.byteLength;
    }

    const smallBuffer = immutableStorageBuffer(
      device,
      new Uint8Array(smallValues.buffer),
      "Limite f32 small tensors",
    );
    const immutableShards = Object.freeze(shards.slice());
    let destroyed = false;

    return Object.freeze({
      manifest,
      shards: immutableShards,
      smallWeights: smallBuffer,
      smallValues,
      smallLayout: SMALL_TENSOR_LAYOUT,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        for (const shard of immutableShards) shard.destroy();
        smallBuffer.destroy();
      },
    });
  } catch (error) {
    for (const shard of shards) shard.destroy();
    throw error;
  }
}

async function fetchShard(
  url: string,
  shard: LimiteShardManifest,
  shardIndex: number,
  totalBytes: number,
  previousBytes: number,
  onProgress: LimiteArtifactProgressCallback,
  cache: ArtifactCacheSession | null,
): Promise<Uint8Array<ArrayBuffer>> {
  const cached = await readCachedShard(cache, shard);
  if (cached) {
    onProgress({
      phase: "shards",
      source: "cache",
      loadedBytes: previousBytes + cached.byteLength,
      totalBytes,
      shardIndex,
      shardLoadedBytes: cached.byteLength,
      shardByteLength: shard.byteLength,
    });
    return cached;
  }

  const response = await fetch(url, { cache: ARTIFACT_FETCH_CACHE });
  if (!response.ok) {
    throw new Error(
      `Could not fetch Limite full-BF16 shard ${shardIndex + 1}/${SHARD_COUNT} ` +
        `(${response.status} ${response.statusText}).`,
    );
  }
  const contentLength = response.headers.get("content-length");
  if (contentLength !== null && integerString(contentLength) !== shard.byteLength) {
    throw new Error(
      `Shard ${shard.file} declared ${shard.byteLength} bytes but HTTP returned ${contentLength}.`,
    );
  }
  if (!response.body) throw new Error(`Shard ${shard.file} has no response body.`);

  const bytes = new Uint8Array(shard.byteLength);
  const reader = response.body.getReader();
  let offset = 0;
  onProgress({
    phase: "shards",
    source: "network",
    loadedBytes: previousBytes,
    totalBytes,
    shardIndex,
    shardLoadedBytes: 0,
    shardByteLength: shard.byteLength,
  });

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.byteLength > bytes.byteLength) {
      await reader.cancel();
      throw new Error(`Shard ${shard.file} exceeds its declared byte length.`);
    }
    bytes.set(value, offset);
    offset += value.byteLength;
    onProgress({
      phase: "shards",
      source: "network",
      loadedBytes: previousBytes + offset,
      totalBytes,
      shardIndex,
      shardLoadedBytes: offset,
      shardByteLength: shard.byteLength,
    });
  }

  if (offset !== shard.byteLength) {
    throw new Error(`Shard ${shard.file} is ${offset} bytes; expected ${shard.byteLength}.`);
  }
  await verifyShardHash(shard, bytes);
  await cacheShard(cache, shard, bytes);
  return bytes;
}

function shardCacheName(shard: LimiteShardManifest): string {
  return `kalkulator-model-v${MANIFEST_VERSION}-${shard.sha256.slice(0, 20)}.bin`;
}

async function prepareArtifactCache(
  shards: readonly LimiteShardManifest[],
  totalBytes: number,
): Promise<ArtifactCacheSession | null> {
  if (USING_CUSTOM_DEV_MANIFEST) return null;
  await requestPersistentArtifactStorage();

  let root: FileSystemDirectoryHandle;
  try {
    root = await navigator.storage.getDirectory();
  } catch (error) {
    console.warn("Persistent model cache is unavailable; shards will use the network.", error);
    return null;
  }

  const current = new Map(shards.map(shard => [shardCacheName(shard), shard.byteLength]));
  let cachedBytes = 0;
  try {
    const iterable = root as FileSystemDirectoryHandle & {
      keys(): AsyncIterableIterator<string>;
    };
    for await (const name of iterable.keys()) {
      const expectedBytes = current.get(name);
      if (expectedBytes !== undefined) {
        try {
          const handle = await root.getFileHandle(name);
          const file = await handle.getFile();
          if (file.size === expectedBytes) {
            cachedBytes += expectedBytes;
          } else {
            await root.removeEntry(name);
          }
        } catch (error) {
          if (!isDomException(error, "NotFoundError")) {
            console.warn(`Could not inspect cached model shard ${name}.`, error);
          }
        }
        continue;
      }

      if (KALKULATOR_ARTIFACT_CACHE_NAME.test(name)) {
        try {
          await root.removeEntry(name);
        } catch (error) {
          console.warn(`Could not remove stale model cache entry ${name}.`, error);
        }
      }
    }
  } catch (error) {
    console.warn("Could not prune old model cache entries.", error);
  }

  await reportArtifactCacheCapacity(
    Math.max(0, totalBytes - cachedBytes),
    Math.max(...shards.map(shard => shard.byteLength)),
  );
  return { root, writeEnabled: true };
}

async function requestPersistentArtifactStorage(): Promise<void> {
  try {
    const persistent = await navigator.storage.persist();
    if (!persistent) {
      console.info("Persistent model storage was not granted; cache eviction remains possible.");
    }
  } catch (error) {
    console.warn("Could not request persistent model storage; continuing best-effort.", error);
  }
}

async function reportArtifactCacheCapacity(
  missingBytes: number,
  largestShardBytes: number,
): Promise<void> {
  if (missingBytes === 0) return;

  try {
    const { quota, usage } = await navigator.storage.estimate();
    if (quota === undefined || usage === undefined) return;
    const availableBytes = Math.max(0, quota - usage);
    const transactionSafeBytes = missingBytes + largestShardBytes;
    if (availableBytes < transactionSafeBytes) {
      console.warn(
        "The browser may not have enough quota to persist the complete model; " +
          "inference will continue if a cache write is rejected.",
        { availableBytes, transactionSafeBytes },
      );
    }
  } catch (error) {
    console.warn("Could not estimate model cache capacity; writes will be attempted.", error);
  }
}

async function readCachedShard(
  cache: ArtifactCacheSession | null,
  shard: LimiteShardManifest,
): Promise<Uint8Array<ArrayBuffer> | null> {
  if (!cache) return null;

  try {
    const name = shardCacheName(shard);
    const handle = await cache.root.getFileHandle(name);
    const file = await handle.getFile();
    if (file.size !== shard.byteLength) {
      await cache.root.removeEntry(name);
      return null;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    try {
      await verifyShardHash(shard, bytes);
      return bytes;
    } catch (error) {
      await cache.root.removeEntry(name);
      console.warn(`Discarded corrupt cached shard ${shard.file}.`, error);
      return null;
    }
  } catch (error) {
    if (isDomException(error, "NotFoundError")) return null;
    cache.writeEnabled = false;
    console.warn("Persistent model cache read failed; using the network for this run.", error);
    return null;
  }
}

async function verifyShardHash(
  shard: LimiteShardManifest,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<void> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  const actual = Array.from(digest, byte => byte.toString(16).padStart(2, "0")).join("");
  if (actual !== shard.sha256) {
    throw new Error(`Shard ${shard.file} failed its SHA-256 integrity check.`);
  }
}

async function cacheShard(
  cache: ArtifactCacheSession | null,
  shard: LimiteShardManifest,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<void> {
  if (!cache?.writeEnabled) return;

  const name = shardCacheName(shard);
  let writable: FileSystemWritableFileStream | null = null;
  try {
    const handle = await cache.root.getFileHandle(name, { create: true });
    writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
  } catch (error) {
    if (writable) await writable.abort().catch(() => {});
    await removeIncompleteCachedShard(cache.root, name, shard.byteLength);
    if (isDomException(error, "QuotaExceededError")) {
      cache.writeEnabled = false;
      console.warn(
        "Model cache quota was exceeded; inference will continue without further cache writes.",
        error,
      );
    } else {
      console.warn("Could not persist this model shard; it will be fetched again next time.", error);
    }
  }
}

async function removeIncompleteCachedShard(
  root: FileSystemDirectoryHandle,
  name: string,
  expectedBytes: number,
): Promise<void> {
  try {
    const handle = await root.getFileHandle(name);
    const file = await handle.getFile();
    if (file.size === expectedBytes) return;
    await root.removeEntry(name);
  } catch (error) {
    if (!isDomException(error, "NotFoundError")) {
      console.warn(`Could not clean up incomplete model cache entry ${name}.`, error);
    }
  }
}

function isDomException(error: unknown, name: string): boolean {
  return error instanceof DOMException && error.name === name;
}

function immutableStorageBuffer(device: GPUDevice, bytes: Uint8Array, label: string): GPUBuffer {
  if (bytes.byteLength === 0 || bytes.byteLength % 4 !== 0) {
    throw new Error(`${label} byte length must be a positive multiple of four.`);
  }
  const buffer = device.createBuffer({
    label,
    size: bytes.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(bytes);
  buffer.unmap();
  return buffer;
}

function extractSmallTensors(
  manifest: LimiteFullBf16Manifest,
  shardIndex: number,
  shardBytes: Uint8Array,
  destination: Float32Array,
): void {
  for (let layer = 0; layer < LAYER_COUNT; layer++) {
    copyF32(
      manifest.tensors[`layers.${layer}.xsa_alpha`] as LimiteSmallTensorManifest,
      shardIndex,
      shardBytes,
      destination,
      SMALL_TENSOR_LAYOUT.xsa + layer * QUERY_HEADS,
    );
    copyF32(
      manifest.tensors[`layers.${layer}.lambdas`] as LimiteSmallTensorManifest,
      shardIndex,
      shardBytes,
      destination,
      SMALL_TENSOR_LAYOUT.lambdas + layer * 4,
    );
  }

  copyF32(
    manifest.tensors["mudd.dense1"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.dense1,
  );
  copyF32(
    manifest.tensors["mudd.dense2"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.dense2,
  );
  copyF32(
    manifest.tensors["mudd.dense2_mlp"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.dense2Mlp,
  );
  copyF32(
    manifest.tensors["mudd.bias"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.bias,
  );
  copyF32(
    manifest.tensors["mudd.bias_mlp"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.biasMlp,
  );
  copyF32(
    manifest.tensors["runtime.gelu_bf16"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.geluByBf16,
  );
  copyF32(
    manifest.tensors["runtime.silu_bf16"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.siluByBf16,
  );
  copyF32(
    manifest.tensors["runtime.sigmoid_bf16"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.sigmoidByBf16,
  );
  copyF32(
    manifest.tensors["runtime.sampling_weight_by_bf16"] as LimiteSmallTensorManifest,
    shardIndex,
    shardBytes,
    destination,
    SMALL_TENSOR_LAYOUT.samplingWeightByBf16,
  );
}

function copyF32(
  tensor: LimiteSmallTensorManifest,
  shardIndex: number,
  shardBytes: Uint8Array,
  destination: Float32Array,
  destinationOffset: number,
): void {
  if (tensor.shard !== shardIndex) return;
  const view = new DataView(
    shardBytes.buffer,
    shardBytes.byteOffset + tensor.offset,
    tensor.byteLength,
  );
  const length = tensor.byteLength / 4;
  for (let index = 0; index < length; index++) {
    destination[destinationOffset + index] = view.getFloat32(index * 4, true);
  }
}

function validateSource(value: unknown): void {
  const source = objectAt(value, "manifest.source");
  exactKeys(source, ["repo", "revision", "file", "sha256", "byteLength"], "manifest.source");
  literal(source.repo, "paradigma-inc/limite-1b-violetto", "manifest.source.repo");
  literal(
    source.revision,
    "b1f3d572ccacb6919f4d64c321b70ba034ddaef2",
    "manifest.source.revision",
  );
  literal(source.file, "model.safetensors", "manifest.source.file");
  literal(
    source.sha256,
    "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5",
    "manifest.source.sha256",
  );
  literal(source.byteLength, 2_070_811_520, "manifest.source.byteLength");
}

function validateFormat(value: unknown): void {
  const format = objectAt(value, "manifest.format");
  exactKeys(
    format,
    [
      "alignment",
      "shardMaxBytes",
      "tiedEmbedding",
      "valueEmbedding",
      "bodyPrecision",
    ],
    "manifest.format",
  );
  literal(format.alignment, ALIGNMENT, "manifest.format.alignment");
  literal(format.shardMaxBytes, SHARD_MAX_BYTES, "manifest.format.shardMaxBytes");
  const tiedEmbedding = objectAt(format.tiedEmbedding, "manifest.format.tiedEmbedding");
  exactKeys(
    tiedEmbedding,
    ["dtype", "packing", "matrixLayout", "roles"],
    "manifest.format.tiedEmbedding",
  );
  literal(tiedEmbedding.dtype, "bf16", "manifest.format.tiedEmbedding.dtype");
  literal(
    tiedEmbedding.packing,
    "two_bf16_values_per_u32_little_endian",
    "manifest.format.tiedEmbedding.packing",
  );
  literal(
    tiedEmbedding.matrixLayout,
    "output_major",
    "manifest.format.tiedEmbedding.matrixLayout",
  );
  exactArray(
    tiedEmbedding.roles,
    ["token_embedding", "lm_head"],
    "manifest.format.tiedEmbedding.roles",
  );
  const valueEmbedding = objectAt(format.valueEmbedding, "manifest.format.valueEmbedding");
  exactKeys(
    valueEmbedding,
    ["dtype", "packing", "matrixLayout", "precision", "roles"],
    "manifest.format.valueEmbedding",
  );
  literal(valueEmbedding.dtype, "bf16", "manifest.format.valueEmbedding.dtype");
  literal(
    valueEmbedding.packing,
    "two_bf16_values_per_u32_little_endian",
    "manifest.format.valueEmbedding.packing",
  );
  literal(
    valueEmbedding.matrixLayout,
    "output_major",
    "manifest.format.valueEmbedding.matrixLayout",
  );
  literal(
    valueEmbedding.precision,
    "oracle_exact",
    "manifest.format.valueEmbedding.precision",
  );
  exactArray(
    valueEmbedding.roles,
    ["value_embedding"],
    "manifest.format.valueEmbedding.roles",
  );
  const bodyPrecision = objectAt(format.bodyPrecision, "manifest.format.bodyPrecision");
  exactKeys(
    bodyPrecision,
    [
      "default",
      "exactBf16Layers",
      "exactBf16Projections",
      "packing",
      "matrixLayout",
    ],
    "manifest.format.bodyPrecision",
  );
  literal(
    bodyPrecision.default,
    "bf16",
    "manifest.format.bodyPrecision.default",
  );
  exactArray(
    bodyPrecision.exactBf16Layers,
    XSA_LAYERS,
    "manifest.format.bodyPrecision.exactBf16Layers",
  );
  exactArray(
    bodyPrecision.exactBf16Projections,
    ["qkv", "o", "gate_up", "down"],
    "manifest.format.bodyPrecision.exactBf16Projections",
  );
  literal(
    bodyPrecision.packing,
    "two_bf16_values_per_u32_little_endian",
    "manifest.format.bodyPrecision.packing",
  );
  literal(
    bodyPrecision.matrixLayout,
    "output_major",
    "manifest.format.bodyPrecision.matrixLayout",
  );
}

function validateModel(value: unknown): void {
  const model = objectAt(value, "manifest.model");
  const literals: Readonly<Record<string, string | number | boolean>> = {
    model_type: "limite",
    hidden_size: HIDDEN_SIZE,
    intermediate_size: INTERMEDIATE_SIZE,
    num_hidden_layers: LAYER_COUNT,
    num_attention_heads: QUERY_HEADS,
    num_key_value_heads: KV_HEADS,
    head_dim: HEAD_DIM,
    vocab_size: VOCAB_SIZE,
    padded_vocab_size: VOCAB_SIZE,
    tokenizer_vocab_size: 151_667,
    bos_token_id: 151_643,
    eos_token_id: 151_643,
    pad_token_id: 151_643,
    max_position_embeddings: 131_072,
    sliding_window: 1_024,
    attention_softmax_scale: 0.1,
    global_nope: true,
    rms_norm_has_weight: false,
    rms_norm_eps_mode: "torch_finfo_default",
    qk_norm: "rms_pre_rope",
    rope_base_global: 1_024,
    rope_base_local: 1_024,
    rope_frac: 0.5,
    rope_n_pairs: 32,
    rope_style: "interleaved_pairs_odd_lane_sign_flip",
    tie_word_embeddings: true,
    lm_head_precision_mode: "oracle_exact",
    value_embedding_precision_mode: "oracle_exact",
    body_precision_mode: "full_bf16_oracle_exact",
    ve_dim: HEAD_DIM,
    ve_stored_heads: KV_HEADS,
    ve_gate_channels: 12,
    ve_gate_scale: 2,
    xsa: true,
    xsa_normalize_eps: 0.0001,
    mudd: true,
    mudd_inter: 32,
    mudd_taps: 3,
  };
  for (const [key, expected] of Object.entries(literals)) {
    literal(model[key], expected, `manifest.model.${key}`);
  }
  exactArray(model.global_layers, GLOBAL_LAYERS, "manifest.model.global_layers");
  exactArray(model.ve_layers, VALUE_LAYERS, "manifest.model.ve_layers");
  exactArray(model.xsa_layers, XSA_LAYERS, "manifest.model.xsa_layers");
  exactArray(model.mudd_layers, [24, 47], "manifest.model.mudd_layers");
  exactArray(model.mudd_at, [24, 47], "manifest.model.mudd_at");

  const taps = objectAt(model.mudd_tap_idx, "manifest.model.mudd_tap_idx");
  exactKeys(taps, ["24", "47"], "manifest.model.mudd_tap_idx");
  exactArray(taps["24"], [0, 12, 24], "manifest.model.mudd_tap_idx.24");
  exactArray(taps["47"], [0, 23, 47], "manifest.model.mudd_tap_idx.47");

  const softcap = objectAt(model.softcap_logits, "manifest.model.softcap_logits");
  exactKeys(softcap, ["a", "b", "c", "kind"], "manifest.model.softcap_logits");
  literal(softcap.kind, "sigmoid", "manifest.model.softcap_logits.kind");
  literal(softcap.a, 23, "manifest.model.softcap_logits.a");
  literal(softcap.b, 5, "manifest.model.softcap_logits.b");
  literal(softcap.c, 7.5, "manifest.model.softcap_logits.c");
}

function validateShard(value: unknown, index: number): LimiteShardManifest {
  const path = `manifest.shards[${index}]`;
  const shard = objectAt(value, path);
  exactKeys(shard, ["file", "url", "byteLength", "sha256"], path);
  const expectedFile = `limite-bf16-${String(index).padStart(2, "0")}.bin`;
  literal(shard.file, expectedFile, `${path}.file`);
  literal(shard.url, expectedFile, `${path}.url`);
  const byteLength = positiveInteger(shard.byteLength, `${path}.byteLength`);
  literal(byteLength, EXPECTED_SHARD_BYTE_LENGTHS[index], `${path}.byteLength`);
  if (byteLength > SHARD_MAX_BYTES || byteLength % 4 !== 0) {
    fail(`${path}.byteLength must be a four-byte-aligned value no larger than ${SHARD_MAX_BYTES}`);
  }
  if (typeof shard.sha256 !== "string" || !SHA256_PATTERN.test(shard.sha256)) {
    fail(`${path}.sha256 must be a lowercase SHA-256 digest`);
  }
  return shard as unknown as LimiteShardManifest;
}

function validatePackedBf16Tensor(
  name: string,
  value: unknown,
  expected: ExpectedPackedBf16,
  shards: readonly LimiteShardManifest[],
  rangesByShard: Range[][],
): void {
  const path = `manifest.tensors.${name}`;
  const tensor = objectAt(value, path);
  exactKeys(
    tensor,
    [
      "shard",
      "offset",
      "offsetWords",
      "byteLength",
      "dtype",
      "shape",
      ...Object.keys(expected.extras),
    ],
    path,
  );
  literal(tensor.dtype, "bf16", `${path}.dtype`);
  exactArray(tensor.shape, expected.shape, `${path}.shape`);
  const [rows, columns] = expected.shape;
  if ((rows * columns) % 2 !== 0) fail(`${path}.shape is not packed-u32 aligned`);
  const shard = shardIndexAt(tensor.shard, `${path}.shard`);
  const offset = alignedOffset(tensor.offset, `${path}.offset`);
  literal(tensor.offsetWords, offset / 4, `${path}.offsetWords`);
  const byteLength = rows * columns * 2;
  literal(tensor.byteLength, byteLength, `${path}.byteLength`);
  boundedRange(shard, offset, byteLength, `${name}.bf16`, shards, rangesByShard);
  for (const [key, expectedValue] of Object.entries(expected.extras)) {
    exactJson(tensor[key], expectedValue, `${path}.${key}`);
  }
}

function validateSmallTensor(
  name: string,
  value: unknown,
  expected: ExpectedSmall,
  shards: readonly LimiteShardManifest[],
  rangesByShard: Range[][],
): void {
  const path = `manifest.tensors.${name}`;
  const tensor = objectAt(value, path);
  exactKeys(
    tensor,
    ["shard", "offset", "offsetWords", "byteLength", "dtype", "shape", ...Object.keys(expected.extras)],
    path,
  );
  literal(tensor.dtype, "f32", `${path}.dtype`);
  exactArray(tensor.shape, expected.shape, `${path}.shape`);
  const shard = shardIndexAt(tensor.shard, `${path}.shard`);
  const offset = alignedOffset(tensor.offset, `${path}.offset`);
  literal(tensor.offsetWords, offset / 4, `${path}.offsetWords`);
  const byteLength = product(expected.shape) * 4;
  literal(tensor.byteLength, byteLength, `${path}.byteLength`);
  boundedRange(shard, offset, byteLength, name, shards, rangesByShard);
  for (const [key, expectedValue] of Object.entries(expected.extras)) {
    exactJson(tensor[key], expectedValue, `${path}.${key}`);
  }
}

function boundedRange(
  shard: number,
  start: number,
  byteLength: number,
  label: string,
  shards: readonly LimiteShardManifest[],
  rangesByShard: Range[][],
): void {
  const end = start + byteLength;
  if (!Number.isSafeInteger(end) || end > shards[shard].byteLength) {
    fail(`${label} extends beyond shard ${shard}`);
  }
  rangesByShard[shard].push({ start, end, label });
}

function shardIndexAt(value: unknown, path: string): number {
  const index = nonNegativeInteger(value, path);
  if (index >= SHARD_COUNT) fail(`${path} must be between 0 and ${SHARD_COUNT - 1}`);
  return index;
}

function alignedOffset(value: unknown, path: string): number {
  const offset = nonNegativeInteger(value, path);
  if (offset % ALIGNMENT !== 0) fail(`${path} must be ${ALIGNMENT}-byte aligned`);
  return offset;
}

function alignOffset(value: number): number {
  return Math.ceil(value / ALIGNMENT) * ALIGNMENT;
}

function objectAt(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    fail(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function arrayAt(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) fail(`${path} must be an array`);
  return value;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[], path: string): void {
  const actual = Object.keys(value).sort();
  const sortedExpected = [...expected].sort();
  if (
    actual.length !== sortedExpected.length ||
    actual.some((key, index) => key !== sortedExpected[index])
  ) {
    fail(`${path} keys must be exactly [${sortedExpected.join(", ")}]`);
  }
}

function literal<T extends string | number | boolean>(value: unknown, expected: T, path: string): void {
  if (value !== expected) fail(`${path} must be ${JSON.stringify(expected)}`);
}

function exactArray(value: unknown, expected: readonly unknown[], path: string): void {
  const actual = arrayAt(value, path);
  if (actual.length !== expected.length) fail(`${path} must have length ${expected.length}`);
  for (let index = 0; index < expected.length; index++) {
    if (actual[index] !== expected[index]) {
      fail(`${path}[${index}] must be ${JSON.stringify(expected[index])}`);
    }
  }
}

function exactJson(value: unknown, expected: unknown, path: string): void {
  if (!jsonEqual(value, expected)) {
    fail(`${path} does not match the packed model contract`);
  }
}

function jsonEqual(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => jsonEqual(value, right[index]))
    );
  }
  if (
    typeof left !== "object" ||
    left === null ||
    typeof right !== "object" ||
    right === null
  ) {
    return false;
  }
  const leftObject = left as Record<string, unknown>;
  const rightObject = right as Record<string, unknown>;
  const leftKeys = Object.keys(leftObject).sort();
  const rightKeys = Object.keys(rightObject).sort();
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every(
      (key, index) =>
        key === rightKeys[index] && jsonEqual(leftObject[key], rightObject[key]),
    )
  );
}

function positiveInteger(value: unknown, path: string): number {
  const integer = nonNegativeInteger(value, path);
  if (integer === 0) fail(`${path} must be positive`);
  return integer;
}

function nonNegativeInteger(value: unknown, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(`${path} must be a non-negative safe integer`);
  }
  return value;
}

function integerString(value: string): number {
  if (!/^(0|[1-9][0-9]*)$/.test(value)) return Number.NaN;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : Number.NaN;
}

function product(shape: readonly number[]): number {
  return shape.reduce((size, dimension) => size * dimension, 1);
}

function deepFreeze<T>(value: T): T {
  if (typeof value !== "object" || value === null || Object.isFrozen(value)) return value;
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  return Object.freeze(value);
}

function fail(message: string): never {
  throw new Error(`Invalid Limite full-BF16 manifest: ${message}.`);
}
