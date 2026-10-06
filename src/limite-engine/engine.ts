import {
  LIMITE_CONTEXT_TOKENS,
  LIMITE_DECODE_BATCH_SIZE as DECODE_BATCH_SIZE,
  LIMITE_TEMPERATURE,
  LIMITE_TOKENIZER_VOCAB_SIZE,
  LIMITE_TOP_P,
} from "../limite-config";
import {
  loadLimiteQ4Artifact,
  type LimiteArtifactProgressCallback,
  type LoadedLimiteQ4Artifact,
  type LimiteQ4TensorManifest,
} from "./artifact";
import {
  ATTENTION_PARTIAL_STRIDE,
  ATTENTION_PARTITION_KEYS,
  ELEMENTWISE_WORKGROUP_SIZE,
  GQA_ATTENTION_WGSL,
  LIMITE_HEAD_DIM,
  LIMITE_HIDDEN_SIZE,
  LIMITE_INTERMEDIATE_SIZE,
  LIMITE_KV_HEADS,
  LIMITE_LOCAL_KEY_COUNT,
  LIMITE_QUERY_HEADS,
  LIMITE_RMS_EPSILON,
  LIMITE_VOCAB_SIZE,
  Q4_32_EMBEDDING_WGSL,
  Q4_32_EMBED_WORKGROUP_SIZE,
  QKV_POSTPROCESS_WGSL,
  SWIGLU_WGSL,
} from "./ops";
import {
  GPU_TOP_P_SAMPLER_WGSL,
  SAMPLER_ENTRIES_PER_PARTITION,
  SAMPLER_WORKGROUP_SIZE,
} from "./sampler";
import {
  COPY_VECTOR_WGSL,
  MIX_AND_NORM_WGSL,
  MUDD_WGSL,
  VECTOR_WORKGROUP_SIZE,
} from "./runtime-shaders";
import {
  Q4_32_MATVEC_ROWS_PER_GROUP,
  Q4_32_MATVEC_WGSL,
  RMS_NORM_F32_WGSL,
} from "./shaders";

const LAYER_COUNT = 48;
const KV_WIDTH = LIMITE_KV_HEADS * LIMITE_HEAD_DIM;
const INITIAL_CACHE_TOKENS = 4_096;
const KV_BYTES_PER_TOKEN = KV_WIDTH * 2;
const ROPE_VALUES_PER_TOKEN = 64 * 2;
const ROPE_FREQUENCIES = Array.from({ length: 32 }, (_, pair) => Math.pow(1_024, -pair / 31));
const QKV_OUTPUT_SIZE =
  LIMITE_QUERY_HEADS * LIMITE_HEAD_DIM + 2 * KV_WIDTH + LIMITE_QUERY_HEADS + LIMITE_KV_HEADS;
const GLOBAL_LAYERS = new Set(Array.from({ length: 12 }, (_, index) => index * 4 + 3));
const VALUE_LAYERS = new Set(Array.from({ length: 16 }, (_, index) => index * 3 + 1));
const MAX_ATTENTION_PARTITIONS = Math.ceil(LIMITE_CONTEXT_TOKENS / ATTENTION_PARTITION_KEYS);
const SAMPLER_PARTITIONS = Math.ceil(LIMITE_TOKENIZER_VOCAB_SIZE / SAMPLER_WORKGROUP_SIZE);
const SAMPLER_CANDIDATES = SAMPLER_PARTITIONS * SAMPLER_ENTRIES_PER_PARTITION;
const PARAMETER_BLOCK_BYTES = 256;
const PARAMETER_LAYER_BYTES = PARAMETER_BLOCK_BYTES * 2;
const PARAMETER_SLOT_BYTES = LAYER_COUNT * PARAMETER_LAYER_BYTES;

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;

export type LimiteEngineTimings = {
  readonly deviceMs: number;
  readonly artifactMs: number;
  readonly pipelineMs: number;
};

type Q4Operation = {
  readonly bindGroup: GPUBindGroup;
  readonly rows: number;
};

type LayerRuntime = {
  readonly index: number;
  readonly isGlobal: boolean;
  cacheCapacity: number;
  keyCache: GPUBuffer;
  valueCache: GPUBuffer;
  readonly qkv: Q4Operation;
  readonly output: Q4Operation;
  readonly gateUp: Q4Operation;
  readonly down: Q4Operation;
  qkvBindGroups: readonly GPUBindGroup[];
  attentionPartitionBindGroups: readonly GPUBindGroup[];
  attentionFinalizeBindGroups: readonly GPUBindGroup[];
  readonly attentionResidualBindGroup: GPUBindGroup;
  readonly mlpResidualBindGroup: GPUBindGroup;
};

type Pipelines = {
  readonly embedding: GPUComputePipeline;
  readonly rms: GPUComputePipeline;
  readonly q4: GPUComputePipeline;
  readonly qkv: GPUComputePipeline;
  readonly attentionPartition: GPUComputePipeline;
  readonly attentionFinalize: GPUComputePipeline;
  readonly mixAndNorm: GPUComputePipeline;
  readonly swiglu: GPUComputePipeline;
  readonly copy: GPUComputePipeline;
  readonly mudd: GPUComputePipeline;
  readonly samplePartitions: GPUComputePipeline;
  readonly sampleTop: GPUComputePipeline;
};

type Scratch = {
  readonly token: GPUBuffer;
  readonly tokenReadback: GPUBuffer;
  readonly rng: GPUBuffer;
  readonly rope: GPUBuffer;
  readonly embedding: GPUBuffer;
  readonly hiddenA: GPUBuffer;
  readonly hiddenB: GPUBuffer;
  readonly attentionInput: GPUBuffer;
  readonly valueEmbedding: GPUBuffer;
  readonly history0: GPUBuffer;
  readonly history12: GPUBuffer;
  readonly history23: GPUBuffer;
  readonly muddResidual: GPUBuffer;
  readonly projectedQkv: GPUBuffer;
  readonly query: GPUBuffer;
  readonly attentionPartials: GPUBuffer;
  readonly attentionOutput: GPUBuffer;
  readonly projectedOutput: GPUBuffer;
  readonly mixed: GPUBuffer;
  readonly mlpInput: GPUBuffer;
  readonly gateUp: GPUBuffer;
  readonly activated: GPUBuffer;
  readonly projectedDown: GPUBuffer;
  readonly logits: GPUBuffer;
  readonly candidateValues: GPUBuffer;
  readonly candidateIds: GPUBuffer;
  readonly prefixMasses: GPUBuffer;
};

type StaticBindings = {
  readonly embedding: GPUBindGroup;
  readonly valueEmbedding: GPUBindGroup;
  readonly embeddingNorm: GPUBindGroup;
  readonly inputNorm: GPUBindGroup;
  readonly copyHistory0: GPUBindGroup;
  readonly copyHistory12: GPUBindGroup;
  readonly copyHistory23: GPUBindGroup;
  readonly swiglu: GPUBindGroup;
  readonly mudd24: GPUBindGroup;
  readonly mudd47: GPUBindGroup;
  readonly head: Q4Operation;
  readonly samplerPartitions: GPUBindGroup;
  readonly samplerTop: GPUBindGroup;
};

export class LimiteWebGpuEngine {
  readonly device: GPUDevice;
  readonly timings: LimiteEngineTimings;

  readonly #artifact: LoadedLimiteQ4Artifact;
  readonly #pipelines: Pipelines;
  readonly #scratch: Scratch;
  readonly #parameters: GPUBuffer;
  readonly #parameterValues = new Uint32Array(
    (PARAMETER_SLOT_BYTES * DECODE_BATCH_SIZE) / Uint32Array.BYTES_PER_ELEMENT,
  );
  readonly #bindings: StaticBindings;
  readonly #layers: readonly LayerRuntime[];
  readonly #ropeValues = new Float32Array(DECODE_BATCH_SIZE * ROPE_VALUES_PER_TOKEN);
  #position = 0;

  private constructor(
    device: GPUDevice,
    artifact: LoadedLimiteQ4Artifact,
    pipelines: Pipelines,
    scratch: Scratch,
    parameters: GPUBuffer,
    bindings: StaticBindings,
    layers: readonly LayerRuntime[],
    timings: LimiteEngineTimings,
  ) {
    this.device = device;
    this.#artifact = artifact;
    this.#pipelines = pipelines;
    this.#scratch = scratch;
    this.#parameters = parameters;
    this.#bindings = bindings;
    this.#layers = layers;
    this.timings = timings;
  }

  static async create(
    onProgress: LimiteArtifactProgressCallback = () => {},
  ): Promise<LimiteWebGpuEngine> {
    if (!navigator.gpu) throw new Error("WebGPU is unavailable in this browser.");
    const deviceStarted = performance.now();
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available.");
    const largestShard = 112 * 1024 * 1024;
    if (
      adapter.limits.maxStorageBufferBindingSize < largestShard ||
      adapter.limits.maxBufferSize < largestShard
    ) {
      throw new Error("This WebGPU device cannot bind the model's compact weight shards.");
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: largestShard,
        maxBufferSize: largestShard,
      },
    });
    device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured WebGPU error", event.error);
    });
    const deviceMs = performance.now() - deviceStarted;

    const artifactStarted = performance.now();
    const artifact = await loadLimiteQ4Artifact(device, onProgress);
    const artifactMs = performance.now() - artifactStarted;
    try {
      const pipelineStarted = performance.now();
      const pipelines = await createPipelines(device);
      const pipelineMs = performance.now() - pipelineStarted;
      device.pushErrorScope("validation");
      const scratch = createScratch(device);
      const parameters = createParameterBuffer(device);
      seedRng(device, scratch.rng);
      const { bindings, layers } = createBindings(
        device,
        artifact,
        pipelines,
        scratch,
        parameters,
      );
      const bindingError = await device.popErrorScope();
      if (bindingError) throw new Error(`WebGPU resource setup failed: ${bindingError.message}`);
      return new LimiteWebGpuEngine(
        device,
        artifact,
        pipelines,
        scratch,
        parameters,
        bindings,
        layers,
        { deviceMs, artifactMs, pipelineMs },
      );
    } catch (error) {
      artifact.destroy();
      device.destroy();
      throw error;
    }
  }

  get position(): number {
    return this.#position;
  }

  reset(): void {
    this.#position = 0;
    seedRng(this.device, this.#scratch.rng);
  }

  async prefill(tokens: readonly number[], onProgress: (processed: number) => void = () => {}): Promise<number> {
    if (tokens.length === 0) throw new Error("The formatted prompt is empty.");
    if (tokens.length >= LIMITE_CONTEXT_TOKENS) {
      throw new Error(`The prompt exceeds the ${LIMITE_CONTEXT_TOKENS}-token context.`);
    }
    this.reset();
    await this.#ensureCache(tokens.length);
    for (let index = 0; index < tokens.length - 1; index++) {
      await this.#submitToken(tokens[index], false);
      if ((index + 1) % 32 === 0) {
        await this.device.queue.onSubmittedWorkDone();
        onProgress(index + 1);
      }
    }
    return this.#submitToken(tokens[tokens.length - 1], true);
  }

  async decodeBatch(token: number, count = DECODE_BATCH_SIZE): Promise<readonly number[]> {
    const remaining = LIMITE_CONTEXT_TOKENS - this.#position;
    if (remaining <= 0) {
      throw new Error(`The ${LIMITE_CONTEXT_TOKENS}-token context is full.`);
    }
    const batchCount = Math.min(Math.max(Math.floor(count), 1), DECODE_BATCH_SIZE, remaining);
    const startingPosition = this.#position;
    await this.#ensureCache(startingPosition + batchCount);
    this.device.queue.writeBuffer(this.#scratch.token, 0, new Uint32Array([token]));
    for (let index = 0; index < batchCount; index++) {
      this.#prepareParameterSlot(startingPosition + index, index);
    }
    this.#uploadParameterSlots(batchCount);

    const encoder = this.device.createCommandEncoder({
      label: `Limite tokens ${startingPosition}-${startingPosition + batchCount - 1}`,
    });
    for (let index = 0; index < batchCount; index++) {
      const position = startingPosition + index;
      const pass = encoder.beginComputePass({ label: `Limite decode ${position}` });
      this.#encodeToken(pass, position, index, true);
      pass.end();
      encoder.copyBufferToBuffer(
        this.#scratch.token,
        0,
        this.#scratch.tokenReadback,
        index * Uint32Array.BYTES_PER_ELEMENT,
        Uint32Array.BYTES_PER_ELEMENT,
      );
    }
    this.device.queue.submit([encoder.finish()]);
    this.#position += batchCount;
    await this.#scratch.tokenReadback.mapAsync(GPUMapMode.READ);
    const sampled = Array.from(
      new Uint32Array(this.#scratch.tokenReadback.getMappedRange(), 0, batchCount),
    );
    this.#scratch.tokenReadback.unmap();
    if (sampled.some((value) => value >= LIMITE_VOCAB_SIZE)) {
      throw new Error("The GPU sampler returned an invalid token.");
    }
    return sampled;
  }

  destroy(): void {
    this.#artifact.destroy();
    this.#parameters.destroy();
    for (const buffer of Object.values(this.#scratch)) buffer.destroy();
    this.device.destroy();
  }

  async #ensureCache(required: number): Promise<void> {
    const current = this.#layers[3].cacheCapacity;
    const capacity = Math.min(
      LIMITE_CONTEXT_TOKENS,
      Math.max(INITIAL_CACHE_TOKENS, 2 ** Math.ceil(Math.log2(required))),
    );
    // Retain the current solve's prefix; release oversized caches on a new solve.
    if (capacity === current || (this.#position > 0 && capacity < current)) return;
    const replacements: { layer: LayerRuntime; key: GPUBuffer; value: GPUBuffer }[] = [];
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    let failure: unknown;
    try {
      const encoder = this.device.createCommandEncoder({ label: "Limite grow attention cache" });
      for (const layer of this.#layers) {
        if (!layer.isGlobal) continue;
        const key = createCacheBuffer(this.device, layer.index, "key", capacity);
        const value = createCacheBuffer(this.device, layer.index, "value", capacity);
        replacements.push({ layer, key, value });
        const bytes = this.#position * KV_BYTES_PER_TOKEN;
        if (bytes > 0) {
          encoder.copyBufferToBuffer(layer.keyCache, 0, key, 0, bytes);
          encoder.copyBufferToBuffer(layer.valueCache, 0, value, 0, bytes);
        }
      }
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    } catch (error) {
      failure = error;
    }
    const validation = await this.device.popErrorScope();
    const allocation = await this.device.popErrorScope();
    if (failure || validation || allocation) {
      for (const { key, value } of replacements) {
        key.destroy();
        value.destroy();
      }
      throw new Error("Could not grow GPU memory for this calculation. Try a shorter problem.", {
        cause: failure ?? validation ?? allocation,
      });
    }
    for (const { layer, key, value } of replacements) {
      const bindings = createCacheBindings(
        this.device, this.#pipelines, this.#scratch, this.#parameters,
        this.#artifact.smallWeights, layer.index, key, value,
      );
      layer.keyCache.destroy();
      layer.valueCache.destroy();
      Object.assign(layer, bindings, { keyCache: key, valueCache: value, cacheCapacity: capacity });
    }
  }

  #prepareParameterSlot(position: number, parameterSlot: number): void {
    for (let pair = 0; pair < 32; pair++) {
      const theta = position * ROPE_FREQUENCIES[pair];
      const offset = parameterSlot * ROPE_VALUES_PER_TOKEN + pair * 4;
      const cosine = Math.cos(theta);
      const sine = Math.sin(theta);
      this.#ropeValues[offset] = cosine;
      this.#ropeValues[offset + 1] = sine;
      this.#ropeValues[offset + 2] = cosine;
      this.#ropeValues[offset + 3] = -sine;
    }
    for (const layer of this.#layers) {
      const cacheSlot = layer.isGlobal ? position : position % LIMITE_LOCAL_KEY_COUNT;
      const partitions = layer.isGlobal
        ? Math.ceil((position + 1) / ATTENTION_PARTITION_KEYS)
        : Math.ceil(Math.min(position + 1, LIMITE_LOCAL_KEY_COUNT) / ATTENTION_PARTITION_KEYS);
      const qkvOffset = parameterBlockOffset(parameterSlot, layer.index, false) / 4;
      this.#parameterValues.set(
        [
          position,
          cacheSlot,
          layer.cacheCapacity,
          layer.isGlobal ? 1 : 0,
          VALUE_LAYERS.has(layer.index) ? 1 : 0,
          0,
          0,
          0,
          0,
          0,
          parameterSlot,
          0,
        ],
        qkvOffset,
      );
      const attentionOffset = parameterBlockOffset(parameterSlot, layer.index, true) / 4;
      this.#parameterValues.set(
        [
          position,
          cacheSlot,
          position + 1,
          layer.isGlobal ? 1 : 0,
          partitions,
          MAX_ATTENTION_PARTITIONS,
          0,
          0,
          0,
          0,
          this.#artifact.smallLayout.xsa + layer.index * LIMITE_QUERY_HEADS,
          0,
          0,
          layer.cacheCapacity,
          0,
          0,
        ],
        attentionOffset,
      );
    }
  }

  #uploadParameterSlots(count: number): void {
    this.device.queue.writeBuffer(this.#scratch.rope, 0, this.#ropeValues, 0, count * ROPE_VALUES_PER_TOKEN);
    this.device.queue.writeBuffer(
      this.#parameters,
      0,
      new Uint8Array(this.#parameterValues.buffer, 0, count * PARAMETER_SLOT_BYTES),
    );
  }

  async #submitToken(token: number, sample: boolean): Promise<number> {
    const position = this.#position;
    if (position >= LIMITE_CONTEXT_TOKENS) {
      throw new Error(`The ${LIMITE_CONTEXT_TOKENS}-token context is full.`);
    }

    this.device.queue.writeBuffer(this.#scratch.token, 0, new Uint32Array([token]));
    this.#prepareParameterSlot(position, 0);
    this.#uploadParameterSlots(1);

    const encoder = this.device.createCommandEncoder({ label: `Limite token ${position}` });
    const pass = encoder.beginComputePass({ label: "Limite decode" });

    this.#encodeToken(pass, position, 0, sample);
    pass.end();
    if (sample) {
      encoder.copyBufferToBuffer(this.#scratch.token, 0, this.#scratch.tokenReadback, 0, 4);
    }
    this.device.queue.submit([encoder.finish()]);
    this.#position++;

    if (!sample) {
      return -1;
    }
    await this.#scratch.tokenReadback.mapAsync(GPUMapMode.READ);
    const sampled = new Uint32Array(this.#scratch.tokenReadback.getMappedRange())[0];
    this.#scratch.tokenReadback.unmap();
    if (sampled >= LIMITE_VOCAB_SIZE) {
      throw new Error("The GPU sampler returned an invalid token.");
    }
    return sampled;
  }

  #encodeToken(
    pass: GPUComputePassEncoder,
    position: number,
    parameterSlot: number,
    sample: boolean,
  ): void {
    const keyCount = Math.min(position + 1, LIMITE_LOCAL_KEY_COUNT);
    const localPartitions = Math.ceil(keyCount / ATTENTION_PARTITION_KEYS);
    const globalPartitions = Math.ceil((position + 1) / ATTENTION_PARTITION_KEYS);

    dispatch(pass, this.#pipelines.embedding, this.#bindings.embedding, 5);
    dispatch(pass, this.#pipelines.embedding, this.#bindings.valueEmbedding, 1);
    dispatch(pass, this.#pipelines.rms, this.#bindings.embeddingNorm, 1);
    dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory0, 5);
    dispatch(pass, this.#pipelines.rms, this.#bindings.inputNorm, 1);

    for (let index = 0; index < this.#layers.length; index++) {
      const layer = this.#layers[index];
      if (index === 24) dispatch(pass, this.#pipelines.mudd, this.#bindings.mudd24, 1);
      if (index === 47) dispatch(pass, this.#pipelines.mudd, this.#bindings.mudd47, 1);

      dispatchQ4(pass, this.#pipelines.q4, layer.qkv);
      dispatch(pass, this.#pipelines.qkv, layer.qkvBindGroups[parameterSlot], 12);
      const partitions = layer.isGlobal ? globalPartitions : localPartitions;
      dispatch(
        pass,
        this.#pipelines.attentionPartition,
        layer.attentionPartitionBindGroups[parameterSlot],
        LIMITE_QUERY_HEADS,
        partitions,
      );
      dispatch(
        pass,
        this.#pipelines.attentionFinalize,
        layer.attentionFinalizeBindGroups[parameterSlot],
        LIMITE_QUERY_HEADS,
      );
      dispatchQ4(pass, this.#pipelines.q4, layer.output);
      dispatch(pass, this.#pipelines.mixAndNorm, layer.attentionResidualBindGroup, 1);
      dispatchQ4(pass, this.#pipelines.q4, layer.gateUp);
      dispatch(
        pass,
        this.#pipelines.swiglu,
        this.#bindings.swiglu,
        Math.ceil(LIMITE_INTERMEDIATE_SIZE / ELEMENTWISE_WORKGROUP_SIZE),
      );
      dispatchQ4(pass, this.#pipelines.q4, layer.down);
      dispatch(pass, this.#pipelines.mixAndNorm, layer.mlpResidualBindGroup, 1);

      if (index === 11) dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory12, 5);
      if (index === 22) dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory23, 5);
    }

    if (sample) {
      dispatchQ4(pass, this.#pipelines.q4, this.#bindings.head);
      dispatch(
        pass,
        this.#pipelines.samplePartitions,
        this.#bindings.samplerPartitions,
        SAMPLER_PARTITIONS,
      );
      dispatch(pass, this.#pipelines.sampleTop, this.#bindings.samplerTop, 1);
    }
  }
}

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const modules = {
    embedding: device.createShaderModule({ label: "Limite Q4 embedding", code: Q4_32_EMBEDDING_WGSL }),
    rms: device.createShaderModule({ label: "Limite RMS", code: RMS_NORM_F32_WGSL }),
    q4: device.createShaderModule({ label: "Limite Q4 matvec", code: Q4_32_MATVEC_WGSL }),
    qkv: device.createShaderModule({ label: "Limite QKV postprocess", code: QKV_POSTPROCESS_WGSL }),
    attention: device.createShaderModule({ label: "Limite attention", code: GQA_ATTENTION_WGSL }),
    mixAndNorm: device.createShaderModule({ label: "Limite residual norm", code: MIX_AND_NORM_WGSL }),
    swiglu: device.createShaderModule({ label: "Limite SwiGLU", code: SWIGLU_WGSL }),
    copy: device.createShaderModule({ label: "Limite vector copy", code: COPY_VECTOR_WGSL }),
    mudd: device.createShaderModule({ label: "Limite MUDD", code: MUDD_WGSL }),
    sampler: device.createShaderModule({ label: "Limite sampler", code: GPU_TOP_P_SAMPLER_WGSL }),
  };
  const compilation = await Promise.all(
    Object.entries(modules).map(async ([name, module]) => ({ name, info: await module.getCompilationInfo() })),
  );
  const errors = compilation.flatMap(({ name, info }) =>
    [...info.messages]
      .filter((message) => message.type === "error")
      .map((message) => `${name} ${message.lineNum}:${message.linePos} ${message.message}`),
  );
  if (errors.length) throw new Error(`WebGPU shader compilation failed:\n${errors.join("\n")}`);

  const pipeline = (label: string, module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipelineAsync({ label, layout: "auto", compute: { module, entryPoint } });
  const [
    embedding,
    rms,
    q4,
    qkv,
    attentionPartition,
    attentionFinalize,
    mixAndNorm,
    swiglu,
    copy,
    mudd,
    samplePartitions,
    sampleTop,
  ] = await Promise.all([
    pipeline("Limite Q4 embedding", modules.embedding, "q4_32_embedding"),
    pipeline("Limite RMS", modules.rms, "rms_norm_f32"),
    pipeline("Limite Q4 matvec", modules.q4, "q4_32_matvec"),
    pipeline("Limite QKV postprocess", modules.qkv, "qkv_postprocess"),
    pipeline("Limite attention partitions", modules.attention, "attention_partition"),
    pipeline("Limite attention finalize", modules.attention, "attention_finalize"),
    pipeline("Limite residual norm", modules.mixAndNorm, "mix_and_norm"),
    pipeline("Limite SwiGLU", modules.swiglu, "swiglu"),
    pipeline("Limite vector copy", modules.copy, "copy_vector"),
    pipeline("Limite MUDD", modules.mudd, "mudd"),
    pipeline("Limite sampler partitions", modules.sampler, "prepare_nucleus_partitions"),
    pipeline("Limite sampler final", modules.sampler, "sample_nucleus"),
  ]);
  return {
    embedding,
    rms,
    q4,
    qkv,
    attentionPartition,
    attentionFinalize,
    mixAndNorm,
    swiglu,
    copy,
    mudd,
    samplePartitions,
    sampleTop,
  };
}

function createScratch(device: GPUDevice): Scratch {
  const storage = (label: string, elements: number, extraUsage = 0) =>
    device.createBuffer({ label, size: align4(elements * 4), usage: STORAGE | extraUsage });
  return {
    token: storage("Limite current token", 1, COPY_DST | GPUBufferUsage.COPY_SRC),
    tokenReadback: device.createBuffer({
      label: "Limite sampled token readback",
      size: DECODE_BATCH_SIZE * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    rng: storage("Limite sampler RNG", 1, COPY_DST),
    rope: storage("Limite local RoPE factors", DECODE_BATCH_SIZE * ROPE_VALUES_PER_TOKEN, COPY_DST),
    embedding: storage("Limite embedding", LIMITE_HIDDEN_SIZE),
    hiddenA: storage("Limite hidden A", LIMITE_HIDDEN_SIZE),
    hiddenB: storage("Limite hidden B", LIMITE_HIDDEN_SIZE),
    attentionInput: storage("Limite attention input", LIMITE_HIDDEN_SIZE),
    valueEmbedding: storage("Limite value embedding", KV_WIDTH),
    history0: storage("Limite history 0", LIMITE_HIDDEN_SIZE),
    history12: storage("Limite history 12", LIMITE_HIDDEN_SIZE),
    history23: storage("Limite history 23", LIMITE_HIDDEN_SIZE),
    muddResidual: storage("Limite MUDD residual", LIMITE_HIDDEN_SIZE),
    projectedQkv: storage("Limite projected QKV", QKV_OUTPUT_SIZE),
    query: storage("Limite query", LIMITE_QUERY_HEADS * LIMITE_HEAD_DIM),
    attentionPartials: storage(
      "Limite attention partials",
      LIMITE_QUERY_HEADS * MAX_ATTENTION_PARTITIONS * ATTENTION_PARTIAL_STRIDE,
    ),
    attentionOutput: storage("Limite attention output", LIMITE_HIDDEN_SIZE),
    projectedOutput: storage("Limite projected attention", LIMITE_HIDDEN_SIZE),
    mixed: storage("Limite mixed residual", LIMITE_HIDDEN_SIZE),
    mlpInput: storage("Limite MLP input", LIMITE_HIDDEN_SIZE),
    gateUp: storage("Limite gate and up", LIMITE_INTERMEDIATE_SIZE * 2),
    activated: storage("Limite SwiGLU activation", LIMITE_INTERMEDIATE_SIZE),
    projectedDown: storage("Limite projected down", LIMITE_HIDDEN_SIZE),
    logits: storage("Limite logits", LIMITE_VOCAB_SIZE),
    candidateValues: storage("Limite sampler candidate values", SAMPLER_CANDIDATES),
    candidateIds: storage("Limite sampler candidate ids", SAMPLER_CANDIDATES),
    prefixMasses: storage("Limite sampler prefix masses", SAMPLER_CANDIDATES),
  };
}

function createCacheBuffer(device: GPUDevice, layer: number, kind: string, capacity: number): GPUBuffer {
  return device.createBuffer({
    label: `Limite layer ${layer} ${kind} cache`,
    size: capacity * KV_BYTES_PER_TOKEN,
    usage: STORAGE | COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

function createCacheBindings(
  device: GPUDevice,
  pipelines: Pipelines,
  scratch: Scratch,
  parameters: GPUBuffer,
  smallWeights: GPUBuffer,
  index: number,
  keyCache: GPUBuffer,
  valueCache: GPUBuffer,
): Pick<LayerRuntime, "qkvBindGroups" | "attentionPartitionBindGroups" | "attentionFinalizeBindGroups"> {
  const qkvBindGroups = Array.from({ length: DECODE_BATCH_SIZE }, (_, parameterSlot) =>
    device.createBindGroup({
      label: `Limite layer ${index} QKV bindings ${parameterSlot}`,
      layout: pipelines.qkv.getBindGroupLayout(0),
      entries: [
        storageEntry(0, scratch.projectedQkv),
        storageEntry(1, scratch.valueEmbedding),
        storageEntry(2, scratch.query),
        storageEntry(3, keyCache),
        storageEntry(4, valueCache),
        uniformRangeEntry(
          5,
          parameters,
          parameterBlockOffset(parameterSlot, index, false),
          48,
        ),
        storageEntry(6, scratch.rope),
      ],
    }),
  );
  const attentionPartitionBindGroups = Array.from(
    { length: DECODE_BATCH_SIZE },
    (_, parameterSlot) =>
      device.createBindGroup({
        label: `Limite layer ${index} attention partition bindings ${parameterSlot}`,
        layout: pipelines.attentionPartition.getBindGroupLayout(0),
        entries: [
          storageEntry(0, scratch.query),
          storageEntry(1, keyCache),
          storageEntry(2, valueCache),
          storageEntry(3, scratch.attentionPartials),
          uniformRangeEntry(
            7,
            parameters,
            parameterBlockOffset(parameterSlot, index, true),
            64,
          ),
        ],
      }),
  );
  const attentionFinalizeBindGroups = Array.from(
    { length: DECODE_BATCH_SIZE },
    (_, parameterSlot) =>
      device.createBindGroup({
        label: `Limite layer ${index} attention finalize bindings ${parameterSlot}`,
        layout: pipelines.attentionFinalize.getBindGroupLayout(0),
        entries: [
          storageEntry(2, valueCache),
          storageEntry(3, scratch.attentionPartials),
          storageEntry(4, scratch.projectedQkv),
          storageEntry(5, smallWeights),
          storageEntry(6, scratch.attentionOutput),
          uniformRangeEntry(
            7,
            parameters,
            parameterBlockOffset(parameterSlot, index, true),
            64,
          ),
        ],
      }),
  );
  return { qkvBindGroups, attentionPartitionBindGroups, attentionFinalizeBindGroups };
}

function createParameterBuffer(device: GPUDevice): GPUBuffer {
  return device.createBuffer({
    label: "Limite decode parameters",
    size: PARAMETER_SLOT_BYTES * DECODE_BATCH_SIZE,
    usage: GPUBufferUsage.UNIFORM | COPY_DST,
  });
}

function createBindings(
  device: GPUDevice,
  artifact: LoadedLimiteQ4Artifact,
  pipelines: Pipelines,
  scratch: Scratch,
  parameters: GPUBuffer,
): { bindings: StaticBindings; layers: readonly LayerRuntime[] } {
  const embed = q4Tensor(artifact, "embed_tokens");
  const valueEmbed = q4Tensor(artifact, "value_embeds");
  const embedding = embeddingBindGroup(device, pipelines.embedding, artifact, embed, scratch.token, scratch.embedding);
  const valueEmbedding = embeddingBindGroup(
    device,
    pipelines.embedding,
    artifact,
    valueEmbed,
    scratch.token,
    scratch.valueEmbedding,
  );
  const embeddingNorm = rmsBindGroup(device, pipelines.rms, scratch.embedding, scratch.hiddenA);
  const inputNorm = rmsBindGroup(device, pipelines.rms, scratch.hiddenA, scratch.attentionInput);
  const copyHistory0 = twoStorageBindGroup(device, pipelines.copy, scratch.hiddenA, scratch.history0);
  const copyHistory12 = twoStorageBindGroup(device, pipelines.copy, scratch.hiddenA, scratch.history12);
  const copyHistory23 = twoStorageBindGroup(device, pipelines.copy, scratch.hiddenB, scratch.history23);
  const swigluParams = immutableUniform(
    device,
    new Uint32Array([0, 0, LIMITE_INTERMEDIATE_SIZE, 0, 0, 0, 0, 0]),
    "SwiGLU parameters",
  );
  const swiglu = device.createBindGroup({
    label: "Limite SwiGLU bindings",
    layout: pipelines.swiglu.getBindGroupLayout(0),
    entries: [
      storageEntry(0, scratch.gateUp),
      storageEntry(1, scratch.activated),
      uniformEntry(2, swigluParams),
    ],
  });

  const mudd24 = muddBindGroup(device, pipelines.mudd, artifact, scratch.hiddenA, scratch.history12, 24, scratch);
  const mudd47 = muddBindGroup(device, pipelines.mudd, artifact, scratch.hiddenB, scratch.history23, 47, scratch);
  const layers: LayerRuntime[] = [];
  const small = artifact.smallValues;
  const lambdaBase = artifact.smallLayout.lambdas;

  for (let index = 0; index < LAYER_COUNT; index++) {
    const isGlobal = GLOBAL_LAYERS.has(index);
    const cacheCapacity = isGlobal ? INITIAL_CACHE_TOKENS : LIMITE_LOCAL_KEY_COUNT;
    const keyCache = createCacheBuffer(device, index, "key", cacheCapacity);
    const valueCache = createCacheBuffer(device, index, "value", cacheCapacity);
    const current = index % 2 === 0 ? scratch.hiddenA : scratch.hiddenB;
    const next = index % 2 === 0 ? scratch.hiddenB : scratch.hiddenA;
    const residualBase = index === 24 || index === 47 ? scratch.muddResidual : current;
    const qkv = q4BindGroup(
      device,
      pipelines.q4,
      artifact,
      q4Tensor(artifact, `layers.${index}.qkv`),
      scratch.attentionInput,
      scratch.projectedQkv,
    );
    const output = q4BindGroup(
      device,
      pipelines.q4,
      artifact,
      q4Tensor(artifact, `layers.${index}.o`),
      scratch.attentionOutput,
      scratch.projectedOutput,
    );
    const gateUp = q4BindGroup(
      device,
      pipelines.q4,
      artifact,
      q4Tensor(artifact, `layers.${index}.gate_up`),
      scratch.mlpInput,
      scratch.gateUp,
    );
    const down = q4BindGroup(
      device,
      pipelines.q4,
      artifact,
      q4Tensor(artifact, `layers.${index}.down`),
      scratch.activated,
      scratch.projectedDown,
    );
    const cacheBindings = createCacheBindings(
      device, pipelines, scratch, parameters, artifact.smallWeights, index, keyCache, valueCache,
    );
    const lambdaOffset = lambdaBase + index * 4;
    const attentionResidualBindGroup = mixBindGroup(
      device,
      pipelines.mixAndNorm,
      residualBase,
      scratch.projectedOutput,
      scratch.mixed,
      scratch.mlpInput,
      small[lambdaOffset],
      small[lambdaOffset + 1],
      `Limite layer ${index} attention residual`,
    );
    const mlpResidualBindGroup = mixBindGroup(
      device,
      pipelines.mixAndNorm,
      scratch.mixed,
      scratch.projectedDown,
      next,
      scratch.attentionInput,
      small[lambdaOffset + 2],
      small[lambdaOffset + 3],
      `Limite layer ${index} MLP residual`,
    );
    layers.push({
      index,
      isGlobal,
      cacheCapacity,
      qkv,
      output,
      gateUp,
      down,
      keyCache,
      valueCache,
      ...cacheBindings,
      attentionResidualBindGroup,
      mlpResidualBindGroup,
    });
  }

  const head = q4BindGroup(device, pipelines.q4, artifact, embed, scratch.attentionInput, scratch.logits);
  const samplerParamsBuffer = samplerParams(device);
  const samplerPartitions = device.createBindGroup({
    label: "Limite sampler partition bindings",
    layout: pipelines.samplePartitions.getBindGroupLayout(0),
    entries: [
      storageEntry(0, scratch.logits),
      storageEntry(1, scratch.candidateValues),
      storageEntry(2, scratch.candidateIds),
      storageEntry(3, scratch.prefixMasses),
      uniformEntry(6, samplerParamsBuffer),
    ],
  });
  const samplerTop = device.createBindGroup({
    label: "Limite sampler final bindings",
    layout: pipelines.sampleTop.getBindGroupLayout(0),
    entries: [
      storageEntry(1, scratch.candidateValues),
      storageEntry(2, scratch.candidateIds),
      storageEntry(3, scratch.prefixMasses),
      storageEntry(4, scratch.token),
      storageEntry(5, scratch.rng),
      uniformEntry(6, samplerParamsBuffer),
    ],
  });

  return {
    bindings: {
      embedding,
      valueEmbedding,
      embeddingNorm,
      inputNorm,
      copyHistory0,
      copyHistory12,
      copyHistory23,
      swiglu,
      mudd24,
      mudd47,
      head,
      samplerPartitions,
      samplerTop,
    },
    layers,
  };
}

function q4Tensor(artifact: LoadedLimiteQ4Artifact, name: string): LimiteQ4TensorManifest {
  const tensor = artifact.manifest.tensors[name];
  if (!tensor || tensor.dtype !== "q4_block32") throw new Error(`Missing Q4 tensor ${name}.`);
  return tensor;
}

function q4BindingResource(
  artifact: LoadedLimiteQ4Artifact,
  tensor: LimiteQ4TensorManifest,
): GPUBufferBinding {
  return {
    buffer: artifact.shards[tensor.shard],
    offset: tensor.offset,
    size: align4(tensor.scaleOffset + tensor.scaleByteLength - tensor.offset),
  };
}

function q4BindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  artifact: LoadedLimiteQ4Artifact,
  tensor: LimiteQ4TensorManifest,
  input: GPUBuffer,
  output: GPUBuffer,
): Q4Operation {
  const [rows, columns] = tensor.shape;
  const params = immutableUniform(
    device,
    new Uint32Array([rows, columns, (tensor.scaleOffset - tensor.offset) / 4, 0]),
    "Limite Q4 matvec parameters",
  );
  return {
    rows,
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: q4BindingResource(artifact, tensor) },
        storageEntry(1, input),
        storageEntry(2, output),
        uniformEntry(3, params),
      ],
    }),
  };
}

function embeddingBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  artifact: LoadedLimiteQ4Artifact,
  tensor: LimiteQ4TensorManifest,
  token: GPUBuffer,
  output: GPUBuffer,
): GPUBindGroup {
  const [, width] = tensor.shape;
  const params = immutableUniform(
    device,
    new Uint32Array([
      0,
      0,
      width,
      LIMITE_VOCAB_SIZE,
      (tensor.scaleOffset - tensor.offset) / 4,
      0,
      0,
      0,
    ]),
    "Limite embedding parameters",
  );
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: q4BindingResource(artifact, tensor) },
      storageEntry(1, token),
      storageEntry(2, output),
      uniformEntry(3, params),
    ],
  });
}

function rmsBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  input: GPUBuffer,
  output: GPUBuffer,
): GPUBindGroup {
  const bytes = new ArrayBuffer(32);
  const view = new DataView(bytes);
  view.setUint32(0, 1, true);
  view.setUint32(4, LIMITE_HIDDEN_SIZE, true);
  view.setFloat32(16, LIMITE_RMS_EPSILON, true);
  const params = immutableUniform(device, new Uint8Array(bytes), "Limite RMS parameters");
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [storageEntry(0, input), storageEntry(1, output), uniformEntry(2, params)],
  });
}

function mixBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  left: GPUBuffer,
  right: GPUBuffer,
  mixed: GPUBuffer,
  normalized: GPUBuffer,
  leftScale: number,
  rightScale: number,
  label: string,
): GPUBindGroup {
  const params = immutableUniform(
    device,
    new Float32Array([leftScale, rightScale, 0, 0]),
    `${label} parameters`,
  );
  return device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      storageEntry(0, left),
      storageEntry(1, right),
      storageEntry(2, mixed),
      storageEntry(3, normalized),
      uniformEntry(4, params),
    ],
  });
}

function muddBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  artifact: LoadedLimiteQ4Artifact,
  current: GPUBuffer,
  middle: GPUBuffer,
  layer: 24 | 47,
  scratch: Scratch,
): GPUBindGroup {
  const layout = artifact.smallLayout;
  const params = immutableUniform(
    device,
    new Uint32Array([
      layer,
      layout.dense1,
      layout.dense2,
      layout.dense2Mlp,
      layout.bias,
      layout.biasMlp,
      0,
      0,
    ]),
    `Limite MUDD ${layer} parameters`,
  );
  return device.createBindGroup({
    label: `Limite MUDD ${layer} bindings`,
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      storageEntry(0, artifact.smallWeights),
      storageEntry(1, current),
      storageEntry(2, scratch.history0),
      storageEntry(3, middle),
      storageEntry(4, scratch.attentionInput),
      storageEntry(5, scratch.muddResidual),
      uniformEntry(6, params),
    ],
  });
}

function samplerParams(device: GPUDevice): GPUBuffer {
  const bytes = new ArrayBuffer(48);
  const view = new DataView(bytes);
  view.setUint32(0, LIMITE_TOKENIZER_VOCAB_SIZE, true);
  view.setUint32(12, SAMPLER_CANDIDATES, true);
  view.setUint32(24, SAMPLER_PARTITIONS, true);
  view.setFloat32(32, LIMITE_TEMPERATURE, true);
  view.setFloat32(36, LIMITE_TOP_P, true);
  return immutableUniform(device, new Uint8Array(bytes), "Limite sampler parameters");
}

function immutableUniform(
  device: GPUDevice,
  data: ArrayBufferView<ArrayBuffer>,
  label: string,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size: align16(data.byteLength),
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
  );
  buffer.unmap();
  return buffer;
}

function twoStorageBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  source: GPUBuffer,
  destination: GPUBuffer,
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [storageEntry(0, source), storageEntry(1, destination)],
  });
}

function storageEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer } };
}

function uniformEntry(binding: number, buffer: GPUBuffer): GPUBindGroupEntry {
  return { binding, resource: { buffer } };
}

function uniformRangeEntry(
  binding: number,
  buffer: GPUBuffer,
  offset: number,
  size: number,
): GPUBindGroupEntry {
  return { binding, resource: { buffer, offset, size } };
}

function parameterBlockOffset(
  parameterSlot: number,
  layer: number,
  attention: boolean,
): number {
  return (
    parameterSlot * PARAMETER_SLOT_BYTES +
    layer * PARAMETER_LAYER_BYTES +
    (attention ? PARAMETER_BLOCK_BYTES : 0)
  );
}

function dispatch(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  x: number,
  y = 1,
): void {
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(x, y);
}

function dispatchQ4(
  pass: GPUComputePassEncoder,
  pipeline: GPUComputePipeline,
  operation: Q4Operation,
): void {
  dispatch(pass, pipeline, operation.bindGroup, Math.ceil(operation.rows / Q4_32_MATVEC_ROWS_PER_GROUP));
}

function seedRng(device: GPUDevice, buffer: GPUBuffer): void {
  const seed = new Uint32Array(1);
  crypto.getRandomValues(seed);
  if (seed[0] === 0) seed[0] = 0x6d2b79f5;
  device.queue.writeBuffer(buffer, 0, seed);
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function align16(value: number): number {
  return Math.ceil(value / 16) * 16;
}
