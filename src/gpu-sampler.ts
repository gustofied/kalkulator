import { getWebGPUDevice, numpy as np } from "@jax-js/jax";

const WORKGROUP_SIZE = 256;
const TOP_K = 50;
const PARAM_BYTES = 16;
const LIMITE_HIDDEN_SIZE = 1_280;
const LIMITE_VOCAB_SIZE = 151_680;
const HEAD_GROUPS = Math.ceil(LIMITE_VOCAB_SIZE / WORKGROUP_SIZE);
const HEAD_CANDIDATES = HEAD_GROUPS * TOP_K;

const LM_HEAD_SHADER = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f16>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_ids: array<u32>;

var<workgroup> values: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> ids: array<u32, ${WORKGROUP_SIZE}>;

fn follows(a_value: f32, a_id: u32, b_value: f32, b_id: u32) -> bool {
  return a_value < b_value || (a_value == b_value && a_id > b_id);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let lane = local_id.x;
  let row = group_id.x * ${WORKGROUP_SIZE}u + lane;
  var value = -3.402823e38;
  if (row < ${LIMITE_VOCAB_SIZE}u) {
    let row_start = row * ${LIMITE_HIDDEN_SIZE}u;
    var sum = 0.0;
    for (var index = 0u; index < ${LIMITE_HIDDEN_SIZE / 4}u; index++) {
      let offset = index * 4u;
      sum += hidden[offset] * f32(weights[row_start + offset])
        + hidden[offset + 1u] * f32(weights[row_start + offset + 1u])
        + hidden[offset + 2u] * f32(weights[row_start + offset + 2u])
        + hidden[offset + 3u] * f32(weights[row_start + offset + 3u]);
    }
    value = select(sum, -3.402823e38, sum != sum);
  }
  values[lane] = value;
  ids[lane] = select(row, 0xffffffffu, row >= ${LIMITE_VOCAB_SIZE}u);
  workgroupBarrier();

  var width = 2u;
  while (width <= ${WORKGROUP_SIZE}u) {
    var stride = width >> 1u;
    while (stride > 0u) {
      let other = lane ^ stride;
      if (other > lane) {
        let descending = (lane & width) == 0u;
        let lane_follows = follows(values[lane], ids[lane], values[other], ids[other]);
        let should_swap = select(!lane_follows, lane_follows, descending);
        if (should_swap) {
          let swap_value = values[lane];
          let swap_id = ids[lane];
          values[lane] = values[other];
          ids[lane] = ids[other];
          values[other] = swap_value;
          ids[other] = swap_id;
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    width = width << 1u;
  }

  if (lane < ${TOP_K}u) {
    let output_index = group_id.x * ${TOP_K}u + lane;
    output_values[output_index] = values[lane];
    output_ids[output_index] = ids[lane];
  }
}
`;

const TOP_K_SHADER = /* wgsl */ `
struct Params {
  count: u32,
  use_ids: u32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> input_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_ids: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> values: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> ids: array<u32, ${WORKGROUP_SIZE}>;

fn follows(a_value: f32, a_id: u32, b_value: f32, b_id: u32) -> bool {
  return a_value < b_value || (a_value == b_value && a_id > b_id);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) group_id: vec3<u32>,
) {
  let lane = local_id.x;
  let index = group_id.x * ${WORKGROUP_SIZE}u + lane;
  if (index < params.count) {
    let value = input_values[index];
    values[lane] = select(value, -3.402823e38, value != value);
    if (params.use_ids != 0u) {
      ids[lane] = input_ids[index];
    } else {
      ids[lane] = index;
    }
  } else {
    values[lane] = -3.402823e38;
    ids[lane] = 0xffffffffu;
  }
  workgroupBarrier();

  var width = 2u;
  while (width <= ${WORKGROUP_SIZE}u) {
    var stride = width >> 1u;
    while (stride > 0u) {
      let other = lane ^ stride;
      if (other > lane) {
        let descending = (lane & width) == 0u;
        let lane_follows = follows(values[lane], ids[lane], values[other], ids[other]);
        let should_swap = select(!lane_follows, lane_follows, descending);
        if (should_swap) {
          let value = values[lane];
          let id = ids[lane];
          values[lane] = values[other];
          ids[lane] = ids[other];
          values[other] = value;
          ids[other] = id;
        }
      }
      workgroupBarrier();
      stride = stride >> 1u;
    }
    width = width << 1u;
  }

  if (lane < ${TOP_K}u) {
    let output_index = group_id.x * ${TOP_K}u + lane;
    output_values[output_index] = values[lane];
    output_ids[output_index] = ids[lane];
  }
}
`;

const SAMPLE_SHADER = /* wgsl */ `
struct Params {
  temperature: f32,
  top_p: f32,
  random_value: f32,
  _padding: f32,
}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> selected: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn softcap(logit: f32) -> f32 {
  return 23.0 / (1.0 + exp(-((logit + 5.0) / 7.5)));
}

@compute @workgroup_size(1)
fn main() {
  if (params.temperature <= 0.0) {
    selected[0] = ids[0];
    return;
  }

  var probabilities: array<f32, ${TOP_K}>;
  let maximum = softcap(values[0]);
  var total = 0.0;
  for (var index = 0u; index < ${TOP_K}u; index++) {
    let probability = exp((softcap(values[index]) - maximum) / params.temperature);
    probabilities[index] = probability;
    total += probability;
  }

  var kept_total = 0.0;
  var kept = 0u;
  while (kept < ${TOP_K}u) {
    kept_total += probabilities[kept];
    kept += 1u;
    if (kept_total / total >= params.top_p) {
      break;
    }
  }

  var pick = params.random_value * kept_total;
  for (var index = 0u; index < kept; index++) {
    pick -= probabilities[index];
    if (pick <= 0.0) {
      selected[0] = ids[index];
      return;
    }
  }
  selected[0] = ids[kept - 1u];
}
`;

export class WebGpuSampler {
  readonly #device: GPUDevice;
  readonly #topKPipeline: GPUComputePipeline;
  readonly #samplePipeline: GPUComputePipeline;
  readonly #lmHeadPipeline: GPUComputePipeline | null;
  readonly lmHeadVariant: "scalar" | "unavailable";
  readonly #dummyIds: GPUBuffer;
  readonly #selected: GPUBuffer;
  readonly #readback: GPUBuffer;
  readonly #sampleParams: GPUBuffer;
  readonly #headValues: GPUBuffer;
  readonly #headIds: GPUBuffer;
  readonly #passParams: GPUBuffer[] = [];
  #valueBuffers: GPUBuffer[] = [];
  #idBuffers: GPUBuffer[] = [];
  #candidateCapacity = 0;

  private constructor(
    device: GPUDevice,
    topKPipeline: GPUComputePipeline,
    samplePipeline: GPUComputePipeline,
    lmHeadPipeline: GPUComputePipeline | null,
    lmHeadVariant: "scalar" | "unavailable",
  ) {
    this.#device = device;
    this.#topKPipeline = topKPipeline;
    this.#samplePipeline = samplePipeline;
    this.#lmHeadPipeline = lmHeadPipeline;
    this.lmHeadVariant = lmHeadVariant;
    this.#dummyIds = device.createBuffer({
      label: "kalkulator sampler dummy ids",
      size: 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.#selected = device.createBuffer({
      label: "kalkulator sampled token",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.#readback = device.createBuffer({
      label: "kalkulator sampled token readback",
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.#sampleParams = device.createBuffer({
      label: "kalkulator sampler parameters",
      size: PARAM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.#headValues = device.createBuffer({
      label: "kalkulator lm head candidates",
      size: HEAD_CANDIDATES * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.#headIds = device.createBuffer({
      label: "kalkulator lm head candidate ids",
      size: HEAD_CANDIDATES * 4,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  static async create(): Promise<WebGpuSampler> {
    const device = getWebGPUDevice();
    const topKModule = device.createShaderModule({
      label: "kalkulator hierarchical top-k shader",
      code: TOP_K_SHADER,
    });
    const sampleModule = device.createShaderModule({
      label: "kalkulator top-p sampler shader",
      code: SAMPLE_SHADER,
    });
    const canCompileLmHead =
      device.features.has("shader-f16") &&
      device.limits.maxStorageBufferBindingSize >=
        LIMITE_VOCAB_SIZE * LIMITE_HIDDEN_SIZE * 2;
    const lmHeadVariant = canCompileLmHead ? "scalar" : "unavailable";
    const lmHeadModule = canCompileLmHead
      ? device.createShaderModule({
          label: "kalkulator Limite vocabulary head shader",
          code: LM_HEAD_SHADER,
        })
      : null;
    const compilation = await Promise.all([
      topKModule.getCompilationInfo(),
      sampleModule.getCompilationInfo(),
      ...(lmHeadModule ? [lmHeadModule.getCompilationInfo()] : []),
    ]);
    const errors = compilation.flatMap((info) =>
      [...info.messages]
        .filter((message) => message.type === "error")
        .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`),
    );
    if (errors.length) throw new Error(errors.join("\n"));

    const [topKPipeline, samplePipeline, lmHeadPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: "kalkulator hierarchical top-k",
        layout: "auto",
        compute: {
          module: topKModule,
          entryPoint: "main",
        },
      }),
      device.createComputePipelineAsync({
        label: "kalkulator top-p sampler",
        layout: "auto",
        compute: {
          module: sampleModule,
          entryPoint: "main",
        },
      }),
      lmHeadModule
        ? device.createComputePipelineAsync({
            label: "kalkulator Limite vocabulary head",
            layout: "auto",
            compute: {
              module: lmHeadModule,
              entryPoint: "main",
            },
          })
        : Promise.resolve(null),
    ]);
    return new WebGpuSampler(
      device,
      topKPipeline,
      samplePipeline,
      lmHeadPipeline,
      lmHeadVariant,
    );
  }

  get supportsLmHead(): boolean {
    return this.#lmHeadPipeline !== null;
  }

  async selfTest(): Promise<void> {
    const values = new Float32Array(511).fill(-10);
    values[373] = 20;
    const result = await this.sample(np.array(values), 0, 1);
    if (result !== 373) {
      throw new Error(`GPU sampler self-test returned ${result}, expected 373.`);
    }
  }

  async sample(
    logits: np.Array,
    temperature: number,
    topP: number,
    randomValue: number = Math.random(),
  ): Promise<number> {
    if (logits.ndim !== 1) throw new Error("GPU sampler expects one-dimensional logits.");
    const inputCount = logits.size;
    this.#ensureCapacity(Math.ceil(inputCount / WORKGROUP_SIZE) * TOP_K);
    const inputBuffer = await logits.gpuBuffer();
    const encoder = this.#device.createCommandEncoder({
      label: "kalkulator sample token",
    });

    let count = inputCount;
    let values = inputBuffer;
    let ids = this.#dummyIds;
    let useIds = false;
    let passIndex = 0;

    while (true) {
      const groups = Math.ceil(count / WORKGROUP_SIZE);
      const outputIndex = passIndex % 2;
      const outputValues = this.#valueBuffers[outputIndex];
      const outputIds = this.#idBuffers[outputIndex];
      const params = this.#passParameterBuffer(passIndex);
      this.#device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([count, useIds ? 1 : 0, 0, 0]),
      );

      const pass = encoder.beginComputePass({
        label: `kalkulator top-k pass ${passIndex + 1}`,
      });
      pass.setPipeline(this.#topKPipeline);
      pass.setBindGroup(
        0,
        this.#device.createBindGroup({
          layout: this.#topKPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: values } },
            { binding: 1, resource: { buffer: ids } },
            { binding: 2, resource: { buffer: outputValues } },
            { binding: 3, resource: { buffer: outputIds } },
            { binding: 4, resource: { buffer: params } },
          ],
        }),
      );
      pass.dispatchWorkgroups(groups);
      pass.end();

      count = groups * TOP_K;
      values = outputValues;
      ids = outputIds;
      useIds = true;
      passIndex++;
      if (groups === 1) break;
    }

    const sampleParams = new ArrayBuffer(PARAM_BYTES);
    const sampleFloats = new Float32Array(sampleParams);
    sampleFloats[0] = temperature;
    sampleFloats[1] = topP;
    sampleFloats[2] = randomValue;
    this.#device.queue.writeBuffer(this.#sampleParams, 0, sampleParams);

    const samplePass = encoder.beginComputePass({
      label: "kalkulator nucleus sample",
    });
    samplePass.setPipeline(this.#samplePipeline);
    samplePass.setBindGroup(
      0,
      this.#device.createBindGroup({
        layout: this.#samplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: values } },
          { binding: 1, resource: { buffer: ids } },
          { binding: 2, resource: { buffer: this.#selected } },
          { binding: 3, resource: { buffer: this.#sampleParams } },
        ],
      }),
    );
    samplePass.dispatchWorkgroups(1);
    samplePass.end();
    encoder.copyBufferToBuffer(this.#selected, 0, this.#readback, 0, 4);
    this.#device.queue.submit([encoder.finish()]);

    await this.#readback.mapAsync(GPUMapMode.READ);
    const token = new Uint32Array(this.#readback.getMappedRange().slice(0))[0];
    this.#readback.unmap();
    return token;
  }

  async sampleLmHead(
    hidden: np.Array,
    weight: np.Array,
    temperature: number,
    topP: number,
    randomValue: number = Math.random(),
  ): Promise<number> {
    if (!this.#lmHeadPipeline) {
      throw new Error("Specialized Limite vocabulary head is unavailable.");
    }
    if (
      hidden.dtype !== np.float32 ||
      hidden.size !== LIMITE_HIDDEN_SIZE ||
      weight.dtype !== np.float16 ||
      weight.shape[0] !== LIMITE_VOCAB_SIZE ||
      weight.shape[1] !== LIMITE_HIDDEN_SIZE
    ) {
      throw new Error("Unexpected Limite vocabulary-head tensor shape.");
    }

    this.#ensureCapacity(HEAD_CANDIDATES);
    const [hiddenBuffer, weightBuffer] = await Promise.all([
      hidden.gpuBuffer(),
      weight.gpuBuffer(),
    ]);
    const encoder = this.#device.createCommandEncoder({
      label: "kalkulator fused lm head and sample token",
    });
    const headPass = encoder.beginComputePass({
      label: "kalkulator Limite vocabulary head top-k",
    });
    headPass.setPipeline(this.#lmHeadPipeline);
    headPass.setBindGroup(
      0,
      this.#device.createBindGroup({
        layout: this.#lmHeadPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: hiddenBuffer } },
          { binding: 1, resource: { buffer: weightBuffer } },
          { binding: 2, resource: { buffer: this.#headValues } },
          { binding: 3, resource: { buffer: this.#headIds } },
        ],
      }),
    );
    headPass.dispatchWorkgroups(HEAD_GROUPS);
    headPass.end();

    this.#encodeCandidateSampling(
      encoder,
      this.#headValues,
      this.#headIds,
      HEAD_CANDIDATES,
      temperature,
      topP,
      randomValue,
    );
    this.#device.queue.submit([encoder.finish()]);
    await this.#readback.mapAsync(GPUMapMode.READ);
    const token = new Uint32Array(this.#readback.getMappedRange().slice(0))[0];
    this.#readback.unmap();
    return token;
  }

  #encodeCandidateSampling(
    encoder: GPUCommandEncoder,
    initialValues: GPUBuffer,
    initialIds: GPUBuffer,
    initialCount: number,
    temperature: number,
    topP: number,
    randomValue: number,
  ): void {
    let count = initialCount;
    let values = initialValues;
    let ids = initialIds;
    let passIndex = 0;

    while (true) {
      const groups = Math.ceil(count / WORKGROUP_SIZE);
      const outputIndex = passIndex % 2;
      const outputValues = this.#valueBuffers[outputIndex];
      const outputIds = this.#idBuffers[outputIndex];
      const params = this.#passParameterBuffer(passIndex);
      this.#device.queue.writeBuffer(
        params,
        0,
        new Uint32Array([count, 1, 0, 0]),
      );
      const pass = encoder.beginComputePass({
        label: `kalkulator lm head top-k pass ${passIndex + 1}`,
      });
      pass.setPipeline(this.#topKPipeline);
      pass.setBindGroup(
        0,
        this.#device.createBindGroup({
          layout: this.#topKPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: values } },
            { binding: 1, resource: { buffer: ids } },
            { binding: 2, resource: { buffer: outputValues } },
            { binding: 3, resource: { buffer: outputIds } },
            { binding: 4, resource: { buffer: params } },
          ],
        }),
      );
      pass.dispatchWorkgroups(groups);
      pass.end();
      count = groups * TOP_K;
      values = outputValues;
      ids = outputIds;
      passIndex++;
      if (groups === 1) break;
    }

    const sampleParams = new Float32Array(4);
    sampleParams[0] = temperature;
    sampleParams[1] = topP;
    sampleParams[2] = randomValue;
    this.#device.queue.writeBuffer(this.#sampleParams, 0, sampleParams);
    const samplePass = encoder.beginComputePass({
      label: "kalkulator lm head nucleus sample",
    });
    samplePass.setPipeline(this.#samplePipeline);
    samplePass.setBindGroup(
      0,
      this.#device.createBindGroup({
        layout: this.#samplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: values } },
          { binding: 1, resource: { buffer: ids } },
          { binding: 2, resource: { buffer: this.#selected } },
          { binding: 3, resource: { buffer: this.#sampleParams } },
        ],
      }),
    );
    samplePass.dispatchWorkgroups(1);
    samplePass.end();
    encoder.copyBufferToBuffer(this.#selected, 0, this.#readback, 0, 4);
  }

  #ensureCapacity(candidateCapacity: number): void {
    if (candidateCapacity <= this.#candidateCapacity) return;
    for (const buffer of [...this.#valueBuffers, ...this.#idBuffers]) buffer.destroy();
    const size = Math.max(4, candidateCapacity * 4);
    this.#valueBuffers = [0, 1].map((index) =>
      this.#device.createBuffer({
        label: `kalkulator top-k values ${index}`,
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
    );
    this.#idBuffers = [0, 1].map((index) =>
      this.#device.createBuffer({
        label: `kalkulator top-k ids ${index}`,
        size,
        usage: GPUBufferUsage.STORAGE,
      }),
    );
    this.#candidateCapacity = candidateCapacity;
  }

  #passParameterBuffer(index: number): GPUBuffer {
    while (this.#passParams.length <= index) {
      this.#passParams.push(
        this.#device.createBuffer({
          label: `kalkulator top-k parameters ${this.#passParams.length}`,
          size: PARAM_BYTES,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
      );
    }
    return this.#passParams[index];
  }
}
