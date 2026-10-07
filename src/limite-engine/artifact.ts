import { LIMITE_MANIFEST_SHA256, LIMITE_MANIFEST_URL } from "../limite-config";
import { loadModelMetadata } from "../model-metadata";

const MANIFEST_VERSION = 1;
const SHARD_COUNT = 6;
const ALIGNMENT = 256;
const SHARD_MAX_BYTES = 112 * 1024 * 1024;
const Q4_BLOCK_SIZE = 32;
const LEGACY_MODEL_CACHE_NAME = /^kalkulator-model-v\d+-[0-9a-f]{20}\.bin$/;
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

export type LimiteQ4SourceManifest = {
  readonly repo: "paradigma-inc/limite-1b-violetto";
  readonly revision: "b1f3d572ccacb6919f4d64c321b70ba034ddaef2";
  readonly file: "model.safetensors";
  readonly sha256: "9dc7e89e30acf473629f98ac3ade535fb05cc044fe7e399b78d0bc957283a5f5";
  readonly byteLength: 2_070_811_520;
};

export type LimiteQ4FormatManifest = {
  readonly alignment: 256;
  readonly shardMaxBytes: 117_440_512;
  readonly quantization: {
    readonly dtype: "q4_block32";
    readonly blockSize: 32;
    readonly nibbleOrder: "low_first";
    readonly storedValue: "signed_plus_8";
    readonly signedRange: readonly [-7, 7];
    readonly scale: "max_abs_div_7";
    readonly scaleDtype: "f16";
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

export type LimiteQ4TensorManifest = LimiteTensorBase & {
  readonly dtype: "q4_block32";
  readonly shape: readonly [number, number];
  readonly scaleOffset: number;
  readonly scaleOffsetWords: number;
  readonly scaleByteLength: number;
  readonly scaleShape: readonly [number, number];
  readonly roles?: readonly string[];
  readonly rowShape?: readonly number[];
  readonly parts?: readonly LimiteTensorPart[];
};

export type LimiteSmallTensorManifest = LimiteTensorBase & {
  readonly dtype: "f32";
  readonly transform?: "tanh";
  readonly parts?: readonly string[];
};

export type LimiteTensorManifest = LimiteQ4TensorManifest | LimiteSmallTensorManifest;

export type LimiteQ4Manifest = {
  readonly version: 1;
  readonly source: LimiteQ4SourceManifest;
  readonly format: LimiteQ4FormatManifest;
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
    };

export type LimiteArtifactProgressCallback = (progress: LimiteArtifactProgress) => void;

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
const SMALL_TENSOR_ELEMENTS = MUDD_BIAS_MLP_OFFSET + MUDD_BIAS_MLP_LENGTH;

export const SMALL_TENSOR_LAYOUT = {
  xsa: XSA_OFFSET,
  lambdas: LAMBDAS_OFFSET,
  dense1: MUDD_DENSE1_OFFSET,
  dense2: MUDD_DENSE2_OFFSET,
  dense2Mlp: MUDD_DENSE2_MLP_OFFSET,
  bias: MUDD_BIAS_OFFSET,
  biasMlp: MUDD_BIAS_MLP_OFFSET,
} as const;

export type LoadedLimiteQ4Artifact = {
  readonly manifest: LimiteQ4Manifest;
  readonly shards: readonly GPUBuffer[];
  readonly smallWeights: GPUBuffer;
  readonly smallValues: Float32Array;
  readonly smallLayout: typeof SMALL_TENSOR_LAYOUT;
  destroy(): void;
};

type Range = { readonly start: number; readonly end: number; readonly label: string };
type ExpectedQ4 = {
  readonly shape: readonly [number, number];
  readonly extras: Readonly<Record<string, unknown>>;
};
type ExpectedSmall = {
  readonly shape: readonly number[];
  readonly extras: Readonly<Record<string, unknown>>;
};

const EXPECTED_Q4 = new Map<string, ExpectedQ4>();
const EXPECTED_SMALL = new Map<string, ExpectedSmall>();

EXPECTED_Q4.set("embed_tokens", {
  shape: [VOCAB_SIZE, HIDDEN_SIZE],
  extras: { roles: ["token_embedding", "lm_head"] },
});
EXPECTED_Q4.set("value_embeds", {
  shape: [VOCAB_SIZE, KV_HEADS * HEAD_DIM],
  extras: { rowShape: [KV_HEADS, HEAD_DIM] },
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
  EXPECTED_Q4.set(`${prefix}.qkv`, {
    shape: [qkvRows, HIDDEN_SIZE],
    extras: { parts: qkvParts },
  });
  EXPECTED_Q4.set(`${prefix}.o`, {
    shape: [HIDDEN_SIZE, HIDDEN_SIZE],
    extras: {},
  });
  EXPECTED_Q4.set(`${prefix}.gate_up`, {
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
    },
  });
  EXPECTED_Q4.set(`${prefix}.down`, {
    shape: [HIDDEN_SIZE, INTERMEDIATE_SIZE],
    extras: {},
  });
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

/** Validates the complete, fixed artifact contract emitted by pack-limite-q4.py. */
export function validateLimiteQ4Manifest(value: unknown): LimiteQ4Manifest {
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
  const expectedTensorCount = EXPECTED_Q4.size + EXPECTED_SMALL.size;
  if (Object.keys(tensors).length !== expectedTensorCount) {
    fail(
      `manifest.tensors must contain exactly ${expectedTensorCount} tensors; found ${Object.keys(tensors).length}`,
    );
  }

  const rangesByShard = Array.from({ length: SHARD_COUNT }, () => [] as Range[]);
  for (const [name, expected] of EXPECTED_Q4) {
    const tensor = tensors[name];
    if (tensor === undefined) fail(`manifest.tensors is missing ${name}`);
    validateQ4Tensor(name, tensor, expected, shardManifests, rangesByShard);
  }
  for (const [name, expected] of EXPECTED_SMALL) {
    const tensor = tensors[name];
    if (tensor === undefined) fail(`manifest.tensors is missing ${name}`);
    validateSmallTensor(name, tensor, expected, shardManifests, rangesByShard);
  }
  for (const name of Object.keys(tensors)) {
    if (!EXPECTED_Q4.has(name) && !EXPECTED_SMALL.has(name)) {
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

  return deepFreeze(root) as unknown as LimiteQ4Manifest;
}

export async function loadLimiteQ4Artifact(
  device: GPUDevice,
  onProgress: LimiteArtifactProgressCallback = () => {},
): Promise<LoadedLimiteQ4Artifact> {
  onProgress({ phase: "manifest" });
  const manifestBytes = await loadModelMetadata(LIMITE_MANIFEST_URL, LIMITE_MANIFEST_SHA256, "model manifest");
  const manifest = validateLimiteQ4Manifest(JSON.parse(new TextDecoder().decode(manifestBytes)));
  await removeStaleCachedShards(manifest.shards);
  const totalBytes = manifest.shards.reduce((total, shard) => total + shard.byteLength, 0);
  const smallValues = new Float32Array(SMALL_TENSOR_ELEMENTS);
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
      );
      extractSmallTensors(manifest, shardIndex, bytes, smallValues);
      const buffer = immutableStorageBuffer(device, bytes, `Limite Q4 ${shard.file}`);
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
): Promise<Uint8Array<ArrayBuffer>> {
  const cached = await readCachedShard(shard);
  if (cached) {
    onProgress({
      phase: "shards",
      source: "cache",
      loadedBytes: previousBytes + cached.byteLength,
      totalBytes,
    });
    return cached;
  }

  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) {
    throw new Error(
      `Could not fetch Limite Q4 shard ${shardIndex + 1}/${SHARD_COUNT} ` +
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
    });
  }

  if (offset !== shard.byteLength) {
    throw new Error(`Shard ${shard.file} is ${offset} bytes; expected ${shard.byteLength}.`);
  }
  if (!(await shardMatchesHash(shard, bytes))) {
    throw new Error(`Model download is damaged (${shard.file}). Reload to try again.`);
  }
  await cacheShard(shard, bytes);
  return bytes;
}

function shardCacheName(shard: LimiteShardManifest): string {
  return `kalkulator-q4-${shard.sha256.slice(0, 20)}.bin`;
}

async function removeStaleCachedShards(shards: readonly LimiteShardManifest[]): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory();
    const current = new Set(shards.map(shardCacheName));
    const iterable = root as FileSystemDirectoryHandle & {
      keys(): AsyncIterableIterator<string>;
    };
    for await (const name of iterable.keys()) {
      if (
        (name.startsWith("kalkulator-q4-") || LEGACY_MODEL_CACHE_NAME.test(name)) &&
        !current.has(name)
      ) {
        await root.removeEntry(name);
      }
    }
  } catch (error) {
    console.warn("Could not prune old model cache entries.", error);
  }
}

async function readCachedShard(
  shard: LimiteShardManifest,
): Promise<Uint8Array<ArrayBuffer> | null> {
  try {
    const root = await navigator.storage.getDirectory();
    const handle = await root.getFileHandle(shardCacheName(shard));
    const file = await handle.getFile();
    if (file.size !== shard.byteLength) {
      await root.removeEntry(shardCacheName(shard));
      return null;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    if (!(await shardMatchesHash(shard, bytes))) {
      await root.removeEntry(shardCacheName(shard));
      console.warn(`Replacing damaged cached model shard ${shard.file}.`);
      return null;
    }
    return bytes;
  } catch (error) {
    if (error instanceof DOMException && error.name === "NotFoundError") return null;
    console.warn("Persistent model cache is unavailable; using the network cache.", error);
    return null;
  }
}

async function cacheShard(
  shard: LimiteShardManifest,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory();
    const name = shardCacheName(shard);
    const handle = await root.getFileHandle(name, { create: true });
    const writable = await handle.createWritable();
    try {
      await writable.write(bytes);
      await writable.close();
    } catch (error) {
      await writable.abort().catch(() => {});
      await root.removeEntry(name).catch(() => {});
      throw error;
    }
  } catch (error) {
    console.warn("Could not persist this model shard; it will be fetched again next time.", error);
  }
}

async function shardMatchesHash(
  shard: LimiteShardManifest,
  bytes: Uint8Array<ArrayBuffer>,
): Promise<boolean> {
  const hash = await crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, "0")).join("");
  return hex === shard.sha256;
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
  manifest: LimiteQ4Manifest,
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
  exactKeys(format, ["alignment", "shardMaxBytes", "quantization"], "manifest.format");
  literal(format.alignment, ALIGNMENT, "manifest.format.alignment");
  literal(format.shardMaxBytes, SHARD_MAX_BYTES, "manifest.format.shardMaxBytes");
  const quantization = objectAt(format.quantization, "manifest.format.quantization");
  exactKeys(
    quantization,
    [
      "dtype",
      "blockSize",
      "nibbleOrder",
      "storedValue",
      "signedRange",
      "scale",
      "scaleDtype",
      "matrixLayout",
    ],
    "manifest.format.quantization",
  );
  literal(quantization.dtype, "q4_block32", "manifest.format.quantization.dtype");
  literal(quantization.blockSize, Q4_BLOCK_SIZE, "manifest.format.quantization.blockSize");
  literal(quantization.nibbleOrder, "low_first", "manifest.format.quantization.nibbleOrder");
  literal(
    quantization.storedValue,
    "signed_plus_8",
    "manifest.format.quantization.storedValue",
  );
  exactArray(quantization.signedRange, [-7, 7], "manifest.format.quantization.signedRange");
  literal(quantization.scale, "max_abs_div_7", "manifest.format.quantization.scale");
  literal(quantization.scaleDtype, "f16", "manifest.format.quantization.scaleDtype");
  literal(quantization.matrixLayout, "output_major", "manifest.format.quantization.matrixLayout");
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
  const expectedFile = `limite-q4-${String(index).padStart(2, "0")}.bin`;
  literal(shard.file, expectedFile, `${path}.file`);
  literal(shard.url, expectedFile, `${path}.url`);
  const byteLength = positiveInteger(shard.byteLength, `${path}.byteLength`);
  if (byteLength > SHARD_MAX_BYTES || byteLength % 4 !== 0) {
    fail(`${path}.byteLength must be a four-byte-aligned value no larger than ${SHARD_MAX_BYTES}`);
  }
  if (typeof shard.sha256 !== "string" || !SHA256_PATTERN.test(shard.sha256)) {
    fail(`${path}.sha256 must be a lowercase SHA-256 digest`);
  }
  return shard as unknown as LimiteShardManifest;
}

function validateQ4Tensor(
  name: string,
  value: unknown,
  expected: ExpectedQ4,
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
      "scaleOffset",
      "scaleOffsetWords",
      "scaleByteLength",
      "scaleShape",
      ...Object.keys(expected.extras),
    ],
    path,
  );
  literal(tensor.dtype, "q4_block32", `${path}.dtype`);
  exactArray(tensor.shape, expected.shape, `${path}.shape`);
  const [rows, columns] = expected.shape;
  if (columns % Q4_BLOCK_SIZE !== 0) fail(`${path}.shape is not Q4 block aligned`);
  exactArray(tensor.scaleShape, [rows, columns / Q4_BLOCK_SIZE], `${path}.scaleShape`);

  const shard = shardIndexAt(tensor.shard, `${path}.shard`);
  const offset = alignedOffset(tensor.offset, `${path}.offset`);
  literal(tensor.offsetWords, offset / 4, `${path}.offsetWords`);
  const byteLength = rows * columns / 2;
  literal(tensor.byteLength, byteLength, `${path}.byteLength`);
  const scaleOffset = alignedOffset(tensor.scaleOffset, `${path}.scaleOffset`);
  if (scaleOffset < offset + byteLength) fail(`${path}.scaleOffset overlaps its quant payload`);
  literal(tensor.scaleOffsetWords, scaleOffset / 4, `${path}.scaleOffsetWords`);
  const scaleByteLength = rows * (columns / Q4_BLOCK_SIZE) * 2;
  literal(tensor.scaleByteLength, scaleByteLength, `${path}.scaleByteLength`);
  boundedRange(shard, offset, byteLength, `${name}.quants`, shards, rangesByShard);
  boundedRange(shard, scaleOffset, scaleByteLength, `${name}.scales`, shards, rangesByShard);

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
  throw new Error(`Invalid Limite Q4 manifest: ${message}.`);
}
