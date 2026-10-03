import {
  loadLimiteFullBf16Artifact,
  type LimiteArtifactProgressCallback,
  type LimiteBodyPackedBf16TensorManifest,
  type LimiteEmbeddingPackedBf16TensorManifest,
  type LimitePackedBf16TensorManifest,
  type LoadedLimiteFullBf16Artifact,
} from "./artifact";
import {
  ATTENTION_PARTIAL_STRIDE,
  ATTENTION_PARTITION_KEYS,
  ELEMENTWISE_WORKGROUP_SIZE,
  GPU_TOP_P_SAMPLER_WGSL,
  GQA_ATTENTION_WGSL,
  LIMITE_HEAD_DIM,
  LIMITE_HIDDEN_SIZE,
  LIMITE_INTERMEDIATE_SIZE,
  LIMITE_KV_HEADS,
  LIMITE_LOCAL_KEY_COUNT,
  LIMITE_QUERY_HEADS,
  LIMITE_RMS_EPSILON,
  LIMITE_PADDED_VOCAB_SIZE,
  LIMITE_TOKENIZER_VOCAB_SIZE,
  PACKED_BF16_EMBEDDING_WGSL,
  QKV_POSTPROCESS_WGSL,
  SAMPLER_ENTRIES_PER_PARTITION,
  SAMPLER_WORKGROUP_SIZE,
  SWIGLU_WGSL,
} from "./ops";
import {
  COPY_VECTOR_WGSL,
  MIX_AND_NORM_WGSL,
  MUDD_WGSL,
  VECTOR_WORKGROUP_SIZE,
} from "./runtime-shaders";
import {
  PACKED_BF16_MATVEC_ROWS_PER_GROUP,
  PACKED_BF16_MATVEC_WGSL,
  RMS_NORM_F32_WGSL,
} from "./shaders";

export const LIMITE_CONTEXT_TOKENS = 32_768;
const PINNED_LIMITE_TOKENIZER_URL =
  "https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/tokenizer.json";
export const LIMITE_TOKENIZER_URL = resolveTokenizerUrl();

const LAYER_COUNT = 48;
const KV_WIDTH = LIMITE_KV_HEADS * LIMITE_HEAD_DIM;
const QKV_OUTPUT_SIZE =
  LIMITE_QUERY_HEADS * LIMITE_HEAD_DIM + 2 * KV_WIDTH + LIMITE_QUERY_HEADS + LIMITE_KV_HEADS;
const GLOBAL_LAYERS = new Set(Array.from({ length: 12 }, (_, index) => index * 4 + 3));
const VALUE_LAYERS = new Set(Array.from({ length: 16 }, (_, index) => index * 3 + 1));
const MAX_ATTENTION_PARTITIONS = Math.ceil(LIMITE_CONTEXT_TOKENS / ATTENTION_PARTITION_KEYS);
const SAMPLER_PARTITIONS = Math.ceil(LIMITE_TOKENIZER_VOCAB_SIZE / SAMPLER_WORKGROUP_SIZE);
const SAMPLER_ENTRIES = SAMPLER_PARTITIONS * SAMPLER_ENTRIES_PER_PARTITION;
const DECODE_BATCH_SIZE = 4;
const PARAMETER_BLOCK_BYTES = 256;
const PARAMETER_LAYER_BYTES = PARAMETER_BLOCK_BYTES * 2;
const PARAMETER_SLOT_BYTES = LAYER_COUNT * PARAMETER_LAYER_BYTES;
const LIMITE_SAMPLER_SEED = 0x6d2b79f5;
// CPU-FP32 torch.linspace followed by the checkpoint's power operation. Keeping
// the exact bits avoids the different rounding produced by JS Math.pow.
const LIMITE_ROPE_FREQUENCY_BITS = [
  0x3f800000, 0x3f4cb517, 0x3f23b11d, 0x3f02e4ef, 0x3ed1560c, 0x3ea764a7, 0x3e85da9e,
  0x3e5611cc, 0x3e2b2d9d, 0x3e08e170, 0x3ddae8f2, 0x3daf0c7b, 0x3d8bf9c6, 0x3d5fdc1d,
  0x3d3301c2, 0x3d0f2407, 0x3ce4ebee, 0x3cb70def, 0x3c926096, 0x3c6a190a, 0x3c3b3190,
  0x3c15afe8, 0x3bef641c, 0x3bbf6d21, 0x3b991262, 0x3b74cdd8, 0x3b43c132, 0x3b1c886f,
  0x3afa56ea, 0x3ac82e56, 0x3aa01286, 0x3a800000,
] as const;

const STORAGE = GPUBufferUsage.STORAGE;
const COPY_DST = GPUBufferUsage.COPY_DST;

export type LimiteEngineTimings = {
  readonly deviceMs: number;
  readonly artifactMs: number;
  readonly pipelineMs: number;
};

type MatVecOperation = {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly rows: number;
  readonly rowsPerGroup: number;
};

type LayerRuntime = {
  readonly index: number;
  readonly isGlobal: boolean;
  readonly cacheCapacity: number;
  readonly qkv: MatVecOperation;
  readonly output: MatVecOperation;
  readonly gateUp: MatVecOperation;
  readonly down: MatVecOperation;
  readonly qkvBindGroups: readonly GPUBindGroup[];
  readonly attentionPartitionBindGroups: readonly GPUBindGroup[];
  readonly attentionFinalizeBindGroups: readonly GPUBindGroup[];
  readonly attentionResidualBindGroup: GPUBindGroup;
  readonly mlpResidualBindGroup: GPUBindGroup;
};

type Pipelines = {
  readonly packedBf16Embedding: GPUComputePipeline;
  readonly rms: GPUComputePipeline;
  readonly packedBf16Matvec: GPUComputePipeline;
  readonly qkv: GPUComputePipeline;
  readonly attentionPartition: GPUComputePipeline;
  readonly attentionFinalize: GPUComputePipeline;
  readonly mixAndNorm: GPUComputePipeline;
  readonly swiglu: GPUComputePipeline;
  readonly copy: GPUComputePipeline;
  readonly mudd: GPUComputePipeline;
  readonly prepareNucleusPartitions: GPUComputePipeline;
  readonly sampleNucleus: GPUComputePipeline;
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
  readonly sortedWeights: GPUBuffer;
  readonly sortedTokenIds: GPUBuffer;
  readonly prefixMasses: GPUBuffer;
  readonly samplerParams: GPUBuffer;
};

type StaticBindings = {
  readonly embedding: readonly GPUBindGroup[];
  readonly valueEmbedding: GPUBindGroup;
  readonly embeddingNorm: GPUBindGroup;
  readonly inputNorm: GPUBindGroup;
  readonly copyHistory0: GPUBindGroup;
  readonly copyHistory12: GPUBindGroup;
  readonly copyHistory23: GPUBindGroup;
  readonly swiglu: GPUBindGroup;
  readonly mudd24: GPUBindGroup;
  readonly mudd47: GPUBindGroup;
  readonly head: readonly MatVecOperation[];
  readonly prepareNucleusPartitions: GPUBindGroup;
  readonly sampleNucleus: GPUBindGroup;
};

export class LimiteWebGpuEngine {
  readonly device: GPUDevice;
  readonly timings: LimiteEngineTimings;

  readonly #artifact: LoadedLimiteFullBf16Artifact;
  readonly #pipelines: Pipelines;
  readonly #scratch: Scratch;
  readonly #parameters: GPUBuffer;
  readonly #parameterValues = new Uint32Array(
    (PARAMETER_SLOT_BYTES * DECODE_BATCH_SIZE) / Uint32Array.BYTES_PER_ELEMENT,
  );
  readonly #bindings: StaticBindings;
  readonly #layers: readonly LayerRuntime[];
  #position = 0;

  private constructor(
    device: GPUDevice,
    artifact: LoadedLimiteFullBf16Artifact,
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
    const artifact = await loadLimiteFullBf16Artifact(device, onProgress);
    const artifactMs = performance.now() - artifactStarted;
    try {
      const pipelineStarted = performance.now();
      const pipelines = await createPipelines(device);
      const pipelineMs = performance.now() - pipelineStarted;
      device.pushErrorScope("validation");
      const scratch = createScratch(device, artifact.smallLayout.samplingWeightByBf16);
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

  reset(samplerSeed = LIMITE_SAMPLER_SEED): void {
    this.#position = 0;
    seedRng(this.device, this.#scratch.rng, samplerSeed);
  }

  async prefill(
    tokens: readonly number[],
    samplerSeed = LIMITE_SAMPLER_SEED,
  ): Promise<number> {
    if (tokens.length === 0) throw new Error("The formatted prompt is empty.");
    if (tokens.length >= LIMITE_CONTEXT_TOKENS) {
      throw new Error(`The prompt exceeds the ${LIMITE_CONTEXT_TOKENS}-token context.`);
    }
    this.reset(samplerSeed);
    for (let index = 0; index < tokens.length - 1; index++) {
      this.#submitToken(tokens[index], false);
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
    if (sampled.some((value) => value >= LIMITE_TOKENIZER_VOCAB_SIZE)) {
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

  #prepareParameterSlot(position: number, parameterSlot: number): void {
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
          this.#artifact.smallLayout.sigmoidByBf16,
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
          this.#artifact.smallLayout.sigmoidByBf16,
          0,
        ],
        attentionOffset,
      );
    }
  }

  #uploadParameterSlots(count: number): void {
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
    if (sampled >= LIMITE_TOKENIZER_VOCAB_SIZE) {
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

    for (const embedding of this.#bindings.embedding) {
      dispatch(pass, this.#pipelines.packedBf16Embedding, embedding, 5);
    }
    dispatch(pass, this.#pipelines.packedBf16Embedding, this.#bindings.valueEmbedding, 1);
    dispatch(pass, this.#pipelines.rms, this.#bindings.embeddingNorm, 1);
    dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory0, 5);
    dispatch(pass, this.#pipelines.rms, this.#bindings.inputNorm, 1);

    for (let index = 0; index < this.#layers.length; index++) {
      const layer = this.#layers[index];
      if (index === 24) dispatch(pass, this.#pipelines.mudd, this.#bindings.mudd24, 1);
      if (index === 47) dispatch(pass, this.#pipelines.mudd, this.#bindings.mudd47, 1);

      dispatchMatVec(pass, layer.qkv);
      dispatch(pass, this.#pipelines.qkv, layer.qkvBindGroups[parameterSlot], 12);
      const partitions = layer.isGlobal ? globalPartitions : localPartitions;
      dispatch(
        pass,
        this.#pipelines.attentionPartition,
        layer.attentionPartitionBindGroups[parameterSlot],
        LIMITE_KV_HEADS,
        partitions,
      );
      dispatch(
        pass,
        this.#pipelines.attentionFinalize,
        layer.attentionFinalizeBindGroups[parameterSlot],
        LIMITE_KV_HEADS,
      );
      dispatchMatVec(pass, layer.output);
      dispatch(pass, this.#pipelines.mixAndNorm, layer.attentionResidualBindGroup, 1);
      dispatchMatVec(pass, layer.gateUp);
      dispatch(
        pass,
        this.#pipelines.swiglu,
        this.#bindings.swiglu,
        Math.ceil(LIMITE_INTERMEDIATE_SIZE / ELEMENTWISE_WORKGROUP_SIZE),
      );
      dispatchMatVec(pass, layer.down);
      dispatch(pass, this.#pipelines.mixAndNorm, layer.mlpResidualBindGroup, 1);

      if (index === 11) dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory12, 5);
      if (index === 22) dispatch(pass, this.#pipelines.copy, this.#bindings.copyHistory23, 5);
    }

    if (sample) {
      for (const head of this.#bindings.head) {
        dispatchMatVec(pass, head);
      }
      dispatch(
        pass,
        this.#pipelines.prepareNucleusPartitions,
        this.#bindings.prepareNucleusPartitions,
        SAMPLER_PARTITIONS,
      );
      dispatch(pass, this.#pipelines.sampleNucleus, this.#bindings.sampleNucleus, 1);
    }
  }
}

async function createPipelines(device: GPUDevice): Promise<Pipelines> {
  const modules = {
    packedBf16Embedding: device.createShaderModule({
      label: "Limite exact BF16 embedding",
      code: PACKED_BF16_EMBEDDING_WGSL,
    }),
    rms: device.createShaderModule({ label: "Limite RMS", code: RMS_NORM_F32_WGSL }),
    packedBf16Matvec: device.createShaderModule({
      label: "Limite exact BF16 matvec",
      code: PACKED_BF16_MATVEC_WGSL,
    }),
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

  const pipeline = (
    label: string,
    module: GPUShaderModule,
    entryPoint: string,
    constants?: Record<string, GPUPipelineConstantValue>,
  ) => device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint, ...(constants ? { constants } : {}) },
  });
  const [
    packedBf16Embedding,
    rms,
    packedBf16Matvec,
    qkv,
    attentionPartition,
    attentionFinalize,
    mixAndNorm,
    swiglu,
    copy,
    mudd,
    prepareNucleusPartitions,
    sampleNucleus,
  ] = await Promise.all([
    pipeline(
      "Limite exact BF16 embedding",
      modules.packedBf16Embedding,
      "packed_bf16_embedding",
    ),
    pipeline("Limite RMS", modules.rms, "rms_norm_f32"),
    pipeline("Limite exact BF16 matvec", modules.packedBf16Matvec, "packed_bf16_matvec"),
    pipeline("Limite QKV postprocess", modules.qkv, "qkv_postprocess"),
    pipeline("Limite attention partitions", modules.attention, "attention_partition"),
    pipeline("Limite attention finalize", modules.attention, "attention_finalize"),
    pipeline("Limite residual norm", modules.mixAndNorm, "mix_and_norm"),
    pipeline("Limite SwiGLU", modules.swiglu, "swiglu"),
    pipeline("Limite vector copy", modules.copy, "copy_vector"),
    pipeline("Limite MUDD", modules.mudd, "mudd"),
    pipeline(
      "Limite nucleus partition preparation",
      modules.sampler,
      "prepare_nucleus_partitions",
    ),
    pipeline("Limite nucleus sampling", modules.sampler, "sample_nucleus"),
  ]);
  return {
    packedBf16Embedding,
    rms,
    packedBf16Matvec,
    qkv,
    attentionPartition,
    attentionFinalize,
    mixAndNorm,
    swiglu,
    copy,
    mudd,
    prepareNucleusPartitions,
    sampleNucleus,
  };
}

function createScratch(device: GPUDevice, samplingWeightOffset: number): Scratch {
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
    rope: createRopeTable(device),
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
    logits: storage("Limite logits", LIMITE_PADDED_VOCAB_SIZE),
    sortedWeights: storage("Limite sampler sorted weights", SAMPLER_ENTRIES),
    sortedTokenIds: storage("Limite sampler sorted token ids", SAMPLER_ENTRIES),
    prefixMasses: storage("Limite sampler prefix masses", SAMPLER_ENTRIES),
    samplerParams: samplerParams(device, samplingWeightOffset),
  };
}

function createRopeTable(device: GPUDevice): GPUBuffer {
  const dimensions = 64;
  const valuesPerDimension = 2;
  const buffer = device.createBuffer({
    label: "Limite local RoPE factors",
    size: LIMITE_CONTEXT_TOKENS * dimensions * valuesPerDimension * 4,
    usage: STORAGE,
    mappedAtCreation: true,
  });
  const values = new Float32Array(buffer.getMappedRange());
  const frequencyBits = new Uint32Array(LIMITE_ROPE_FREQUENCY_BITS);
  const frequencies = new Float32Array(frequencyBits.buffer);
  for (let position = 0; position < LIMITE_CONTEXT_TOKENS; position++) {
    const positionBase = position * dimensions * valuesPerDimension;
    for (let dimension = 0; dimension < dimensions; dimension++) {
      const theta = Math.fround(position * frequencies[dimension >> 1]);
      const offset = positionBase + dimension * valuesPerDimension;
      values[offset] = roundToBf16(Math.fround(Math.cos(theta)));
      values[offset + 1] = roundToBf16(
        Math.fround(Math.sin(theta) * (dimension % 2 === 0 ? 1 : -1)),
      );
    }
  }
  buffer.unmap();
  return buffer;
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
  artifact: LoadedLimiteFullBf16Artifact,
  pipelines: Pipelines,
  scratch: Scratch,
  parameters: GPUBuffer,
): { bindings: StaticBindings; layers: readonly LayerRuntime[] } {
  const embed = [
    embeddingPackedBf16Tensor(artifact, "embed_tokens.0"),
    embeddingPackedBf16Tensor(artifact, "embed_tokens.1"),
    embeddingPackedBf16Tensor(artifact, "embed_tokens.2"),
    embeddingPackedBf16Tensor(artifact, "embed_tokens.3"),
  ];
  const valueEmbed = embeddingPackedBf16Tensor(artifact, "value_embeds");
  const embedding = embed.map((tensor, index) =>
    packedBf16EmbeddingBindGroup(
      device,
      pipelines.packedBf16Embedding,
      artifact,
      tensor,
      scratch.token,
      scratch.embedding,
      index === 0,
    ),
  );
  const valueEmbedding = packedBf16EmbeddingBindGroup(
    device,
    pipelines.packedBf16Embedding,
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
    new Uint32Array([
      0,
      0,
      LIMITE_INTERMEDIATE_SIZE,
      artifact.smallLayout.siluByBf16,
      0,
      0,
      0,
      0,
    ]),
    "SwiGLU parameters",
  );
  const swiglu = device.createBindGroup({
    label: "Limite SwiGLU bindings",
    layout: pipelines.swiglu.getBindGroupLayout(0),
    entries: [
      storageEntry(0, scratch.gateUp),
      storageEntry(1, scratch.activated),
      uniformEntry(2, swigluParams),
      storageEntry(3, artifact.smallWeights),
    ],
  });

  const mudd24 = muddBindGroup(device, pipelines.mudd, artifact, scratch.hiddenA, scratch.history12, 24, scratch);
  const mudd47 = muddBindGroup(device, pipelines.mudd, artifact, scratch.hiddenB, scratch.history23, 47, scratch);
  const layers: LayerRuntime[] = [];
  const small = artifact.smallValues;
  const lambdaBase = artifact.smallLayout.lambdas;

  for (let index = 0; index < LAYER_COUNT; index++) {
    const isGlobal = GLOBAL_LAYERS.has(index);
    const cacheCapacity = isGlobal ? LIMITE_CONTEXT_TOKENS : LIMITE_LOCAL_KEY_COUNT;
    const keyCache = device.createBuffer({
      label: `Limite layer ${index} key cache`,
      size: cacheCapacity * KV_WIDTH * Uint16Array.BYTES_PER_ELEMENT,
      usage: STORAGE,
    });
    const valueCache = device.createBuffer({
      label: `Limite layer ${index} value cache`,
      size: cacheCapacity * KV_WIDTH * Uint16Array.BYTES_PER_ELEMENT,
      usage: STORAGE,
    });
    const current = index % 2 === 0 ? scratch.hiddenA : scratch.hiddenB;
    const next = index % 2 === 0 ? scratch.hiddenB : scratch.hiddenA;
    const residualBase = index === 24 || index === 47 ? scratch.muddResidual : current;
    const qkv = bodyMatVecBindGroup(
      device,
      pipelines,
      artifact,
      `layers.${index}.qkv`,
      scratch.attentionInput,
      scratch.projectedQkv,
      0,
    );
    const output = bodyMatVecBindGroup(
      device,
      pipelines,
      artifact,
      `layers.${index}.o`,
      scratch.attentionOutput,
      scratch.projectedOutput,
    );
    const gateUp = bodyMatVecBindGroup(
      device,
      pipelines,
      artifact,
      `layers.${index}.gate_up`,
      scratch.mlpInput,
      scratch.gateUp,
    );
    const down = bodyMatVecBindGroup(
      device,
      pipelines,
      artifact,
      `layers.${index}.down`,
      scratch.activated,
      scratch.projectedDown,
    );
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
          storageEntry(7, artifact.smallWeights),
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
            storageEntry(5, artifact.smallWeights),
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
      qkvBindGroups,
      attentionPartitionBindGroups,
      attentionFinalizeBindGroups,
      attentionResidualBindGroup,
      mlpResidualBindGroup,
    });
  }

  const head = embed.map(tensor =>
    packedBf16MatVecBindGroup(
      device,
      pipelines.packedBf16Matvec,
      artifact,
      tensor,
      scratch.attentionInput,
      scratch.logits,
      tensor.rowStart,
    ),
  );
  const prepareNucleusPartitions = device.createBindGroup({
    label: "Limite nucleus partition bindings",
    layout: pipelines.prepareNucleusPartitions.getBindGroupLayout(0),
    entries: [
      storageEntry(0, scratch.logits),
      storageEntry(1, scratch.sortedWeights),
      storageEntry(2, scratch.sortedTokenIds),
      storageEntry(3, scratch.prefixMasses),
      uniformEntry(6, scratch.samplerParams),
      storageEntry(7, artifact.smallWeights),
    ],
  });
  const sampleNucleus = device.createBindGroup({
    label: "Limite nucleus sampling bindings",
    layout: pipelines.sampleNucleus.getBindGroupLayout(0),
    entries: [
      storageEntry(1, scratch.sortedWeights),
      storageEntry(2, scratch.sortedTokenIds),
      storageEntry(3, scratch.prefixMasses),
      storageEntry(4, scratch.token),
      storageEntry(5, scratch.rng),
      uniformEntry(6, scratch.samplerParams),
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
      prepareNucleusPartitions,
      sampleNucleus,
    },
    layers,
  };
}

function bodyMatrixTensor(
  artifact: LoadedLimiteFullBf16Artifact,
  name: string,
): LimiteBodyPackedBf16TensorManifest {
  const tensor = artifact.manifest.tensors[name];
  if (
    !tensor ||
    tensor.dtype !== "bf16" ||
    !("precision" in tensor) ||
    tensor.precision !== "oracle_exact"
  ) {
    throw new Error(`Missing exact BF16 body tensor ${name}.`);
  }
  return tensor as LimiteBodyPackedBf16TensorManifest;
}

function embeddingPackedBf16Tensor(
  artifact: LoadedLimiteFullBf16Artifact,
  name: string,
): LimiteEmbeddingPackedBf16TensorManifest {
  const tensor = artifact.manifest.tensors[name];
  if (!tensor || tensor.dtype !== "bf16" || !("rowStart" in tensor)) {
    throw new Error(`Missing exact BF16 embedding tensor ${name}.`);
  }
  return tensor as LimiteEmbeddingPackedBf16TensorManifest;
}

function bodyMatVecBindGroup(
  device: GPUDevice,
  pipelines: Pipelines,
  artifact: LoadedLimiteFullBf16Artifact,
  name: string,
  input: GPUBuffer,
  output: GPUBuffer,
  outputOffset = 0,
): MatVecOperation {
  return packedBf16MatVecBindGroup(
    device,
    pipelines.packedBf16Matvec,
    artifact,
    bodyMatrixTensor(artifact, name),
    input,
    output,
    outputOffset,
  );
}
function packedBf16BindingResource(
  artifact: LoadedLimiteFullBf16Artifact,
  tensor: LimitePackedBf16TensorManifest,
): GPUBufferBinding {
  return {
    buffer: artifact.shards[tensor.shard],
    offset: tensor.offset,
    size: align4(tensor.byteLength),
  };
}

function packedBf16MatVecBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  artifact: LoadedLimiteFullBf16Artifact,
  tensor: LimitePackedBf16TensorManifest,
  input: GPUBuffer,
  output: GPUBuffer,
  outputOffset = 0,
): MatVecOperation {
  const [rows, columns] = tensor.shape;
  const params = immutableUniform(
    device,
    new Uint32Array([rows, columns, outputOffset, 0, 0, 0, 0, 0]),
    "Limite packed BF16 matvec parameters",
  );
  return {
    pipeline,
    rows,
    rowsPerGroup: PACKED_BF16_MATVEC_ROWS_PER_GROUP,
    bindGroup: device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: packedBf16BindingResource(artifact, tensor) },
        storageEntry(1, input),
        storageEntry(2, output),
        uniformEntry(3, params),
      ],
    }),
  };
}

function packedBf16EmbeddingBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  artifact: LoadedLimiteFullBf16Artifact,
  tensor: LimiteEmbeddingPackedBf16TensorManifest,
  token: GPUBuffer,
  output: GPUBuffer,
  clearOnMiss = true,
): GPUBindGroup {
  const [rows, width] = tensor.shape;
  const params = immutableUniform(
    device,
    new Uint32Array([
      0,
      0,
      width,
      rows,
      tensor.rowStart,
      clearOnMiss ? 1 : 0,
      0,
      0,
    ]),
    "Limite packed BF16 embedding parameters",
  );
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: packedBf16BindingResource(artifact, tensor) },
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
  artifact: LoadedLimiteFullBf16Artifact,
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
      layout.geluByBf16,
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

function samplerParams(device: GPUDevice, samplingWeightOffset: number): GPUBuffer {
  const bytes = new ArrayBuffer(48);
  const view = new DataView(bytes);
  view.setUint32(0, LIMITE_TOKENIZER_VOCAB_SIZE, true);
  view.setUint32(12, SAMPLER_ENTRIES, true);
  view.setUint32(24, SAMPLER_PARTITIONS, true);
  view.setUint32(28, samplingWeightOffset, true);
  view.setFloat32(32, 0.6, true);
  view.setFloat32(36, 0.95, true);
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

function dispatchMatVec(
  pass: GPUComputePassEncoder,
  operation: MatVecOperation,
): void {
  dispatch(
    pass,
    operation.pipeline,
    operation.bindGroup,
    Math.ceil(operation.rows / operation.rowsPerGroup),
  );
}

function seedRng(
  device: GPUDevice,
  buffer: GPUBuffer,
  seed = LIMITE_SAMPLER_SEED,
): void {
  device.queue.writeBuffer(buffer, 0, new Uint32Array([seed]));
}

const BF16_ROUND_BUFFER = new ArrayBuffer(4);
const BF16_ROUND_F32 = new Float32Array(BF16_ROUND_BUFFER);
const BF16_ROUND_U32 = new Uint32Array(BF16_ROUND_BUFFER);

/** Round f32 to BF16 with round-to-nearest-even, returning it as an f32. */
function roundToBf16(value: number): number {
  BF16_ROUND_F32[0] = Math.fround(value);
  const bits = BF16_ROUND_U32[0];
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0;
  BF16_ROUND_U32[0] = rounded & 0xffff0000;
  return BF16_ROUND_F32[0];
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

function align16(value: number): number {
  return Math.ceil(value / 16) * 16;
}

function resolveTokenizerUrl(): string {
  if (!import.meta.env.DEV) return PINNED_LIMITE_TOKENIZER_URL;
  const configured = import.meta.env.VITE_LIMITE_TOKENIZER_URL?.trim();
  if (!configured) return PINNED_LIMITE_TOKENIZER_URL;
  const url = new URL(configured);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("VITE_LIMITE_TOKENIZER_URL must use HTTP or HTTPS.");
  }
  return url.href;
}
