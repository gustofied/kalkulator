/**
 * Fixed-shape WebGPU decode operators for Limite 1B Violetto.
 *
 * These modules intentionally expose plain WGSL rather than constructing
 * pipelines. The engine owns every mutable buffer and can therefore create
 * pipelines/bind groups once, then encode the same command sequence for every
 * generated token. All element offsets below are scalar offsets, not bytes.
 *
 * Q4_32 artifact layout
 * ----------------------
 * A block represents 32 weights. Signed values are encoded as `quant + 8`,
 * low-index nibble first, packed eight per u32. All code words come first in
 * the matrix buffer, followed by IEEE-f16 block scales packed two per u32.
 * Dequantization is:
 *
 *   weight = scale * (code - 8)
 *
 * Blocks are row-major and rows must have a width divisible by 32. This is the
 * same production tensor layout consumed by Q4_32_MATVEC_WGSL.
 */

export const LIMITE_HIDDEN_SIZE = 1_280;
export const LIMITE_INTERMEDIATE_SIZE = 3_328;
export const LIMITE_QUERY_HEADS = 10;
export const LIMITE_KV_HEADS = 2;
export const LIMITE_HEAD_DIM = 128;
export const LIMITE_GROUP_SIZE = 5;
export const LIMITE_VOCAB_SIZE = 151_680;
export const LIMITE_LOCAL_BACKWARD_SPAN = 1_024;
export const LIMITE_LOCAL_KEY_COUNT = LIMITE_LOCAL_BACKWARD_SPAN + 1;
export const LIMITE_ATTENTION_SCALE = 0.1;
export const LIMITE_RMS_EPSILON = 1 / 128;
export const LIMITE_XSA_EPSILON = 1e-4;

export const Q4_32_BLOCK_ELEMENTS = 32;
export const Q4_32_QUANT_WORDS = 4;
export const Q4_32_EMBED_WORKGROUP_SIZE = 256;

/**
 * Q4_32 embedding gather into an f32 activation vector.
 *
 * Bindings:
 *   0: combined matrix codes and packed f16 scales
 *   1: GPU-resident token ids
 *   2: f32 output activation
 *   3: EmbeddingParams uniform (32 bytes)
 *
 * EmbeddingParams, as eight little-endian u32 values:
 *   [token_offset, output_offset, row_width, vocab_size,
 *    scale_offset_words, 0, 0, 0]
 *
 * Dispatch ceil(row_width / 256) workgroups. For Limite's token embedding,
 * row_width is 1280 and dispatch is exactly (5, 1, 1). The same module gathers
 * the 256-wide value embedding with dispatch (1, 1, 1).
 */
export const Q4_32_EMBEDDING_WGSL = /* wgsl */ `
struct EmbeddingParams {
  token_offset: u32,
  output_offset: u32,
  row_width: u32,
  vocab_size: u32,
  scale_offset_words: u32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: EmbeddingParams;

fn block_scale(block: u32) -> f32 {
  let pair = unpack2x16float(matrix[params.scale_offset_words + (block >> 1u)]);
  return select(pair.x, pair.y, (block & 1u) != 0u);
}

@compute @workgroup_size(${Q4_32_EMBED_WORKGROUP_SIZE})
fn q4_32_embedding(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let column = invocation.x;
  if (column >= params.row_width) {
    return;
  }

  let token = token_ids[params.token_offset];
  if (token >= params.vocab_size) {
    output[params.output_offset + column] = 0.0f;
    return;
  }

  let blocks_per_row = params.row_width / ${Q4_32_BLOCK_ELEMENTS}u;
  let block_in_row = column / ${Q4_32_BLOCK_ELEMENTS}u;
  let element_in_block = column & ${Q4_32_BLOCK_ELEMENTS - 1}u;
  let block = token * blocks_per_row + block_in_row;
  let quant_word = block * ${Q4_32_QUANT_WORDS}u
    + (element_in_block >> 3u);
  let code = (matrix[quant_word] >> ((element_in_block & 7u) * 4u)) & 0x0fu;
  let scale = block_scale(block);
  output[params.output_offset + column] = scale * f32(i32(code) - 8);
}
`;

export const QKV_POSTPROCESS_WORKGROUP_SIZE = LIMITE_HEAD_DIM;

/**
 * Postprocesses one already-projected decode Q/K/V tuple.
 *
 * The projection is f32 and always reserves this fixed layout, including two
 * value-gate slots on layers that do not use value embeddings:
 *
 *   q[1280], k[256], v[256], attention_gate[10], value_gate[2]
 *
 * Query/key RMS is gain-free with epsilon 1/128. Local layers apply the exact
 * Violetto interleaved rotation to the first 64 dimensions; global layers skip
 * RoPE. Value embedding is applied before the K/V cache write. K and V cache
 * buffers use [cache_slot][2][128] within the supplied layer offset. Local
 * layers use the fixed 1025-slot ring `position % 1025`; global layers use the
 * supplied slot.
 *
 * Bindings:
 *   0: projected q/k/v/gates, f32
 *   1: current token value embedding [2][128], f32
 *   2: normalized/rotated query output [10][128], f32
 *   3: layer key cache, f32
 *   4: layer value cache, f32
 *   5: QkvPostprocessParams uniform (48 bytes)
 *   6: precomputed local RoPE cosine/signed-sine pairs, f32
 *
 * Params as u32 values:
 *   [position, slot, cache_capacity, is_global,
 *    has_value_embedding, projected_offset, value_embedding_offset,
 *    query_output_offset, key_cache_offset, value_cache_offset, 0, 0]
 *
 * Dispatch exactly (12, 1, 1): ten query-head workgroups followed by two
 * key/value-head workgroups. Each workgroup has one lane per head dimension.
 */
export const QKV_POSTPROCESS_WGSL = /* wgsl */ `
const QUERY_HEADS = ${LIMITE_QUERY_HEADS}u;
const KV_HEADS = ${LIMITE_KV_HEADS}u;
const HEAD_DIM = ${LIMITE_HEAD_DIM}u;
const Q_SIZE = ${LIMITE_QUERY_HEADS * LIMITE_HEAD_DIM}u;
const KV_SIZE = ${LIMITE_KV_HEADS * LIMITE_HEAD_DIM}u;
const K_OFFSET = Q_SIZE;
const V_OFFSET = Q_SIZE + KV_SIZE;
const ATTENTION_GATE_OFFSET = Q_SIZE + 2u * KV_SIZE;
const VALUE_GATE_OFFSET = ATTENTION_GATE_OFFSET + QUERY_HEADS;
const RMS_EPSILON = ${LIMITE_RMS_EPSILON}f;

struct QkvPostprocessParams {
  position: u32,
  slot: u32,
  cache_capacity: u32,
  is_global: u32,
  has_value_embedding: u32,
  projected_offset: u32,
  value_embedding_offset: u32,
  query_output_offset: u32,
  key_cache_offset: u32,
  value_cache_offset: u32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> value_embedding: array<f32>;
@group(0) @binding(2) var<storage, read_write> query_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<f32>;
@group(0) @binding(5) var<uniform> params: QkvPostprocessParams;
@group(0) @binding(6) var<storage, read> rope_factors: array<f32>;

var<workgroup> squared_sum: array<f32, ${LIMITE_HEAD_DIM}>;

fn sigmoid(value: f32) -> f32 {
  return 1.0f / (1.0f + exp(-value));
}

fn rotate_local(value: f32, paired_value: f32, dimension: u32) -> f32 {
  if (params.is_global != 0u || dimension >= 64u) {
    return value;
  }
  let factor = (params.position * 64u + dimension) * 2u;
  return value * rope_factors[factor]
    + paired_value * rope_factors[factor + 1u];
}

fn reduce_squared_sum(lane: u32, value: f32) -> f32 {
  squared_sum[lane] = value * value;
  workgroupBarrier();
  var stride = HEAD_DIM >> 1u;
  while (stride > 0u) {
    if (lane < stride) {
      squared_sum[lane] += squared_sum[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  return squared_sum[0];
}

fn write_cache_slot() -> u32 {
  if (params.is_global != 0u) {
    return params.slot;
  }
  return params.position % ${LIMITE_LOCAL_KEY_COUNT}u;
}

@compute @workgroup_size(${QKV_POSTPROCESS_WORKGROUP_SIZE})
fn qkv_postprocess(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let head = group.x;
  let cache_slot = write_cache_slot();
  if (head >= QUERY_HEADS + KV_HEADS || cache_slot >= params.cache_capacity) {
    return;
  }

  if (head < QUERY_HEADS) {
    let head_base = params.projected_offset + head * HEAD_DIM;
    let raw = projected[head_base + lane];
    let sum = reduce_squared_sum(lane, raw);
    let inverse_rms = inverseSqrt(sum / f32(HEAD_DIM) + RMS_EPSILON);
    let paired_raw = projected[head_base + (lane ^ 1u)];
    let normalized = raw * inverse_rms;
    let paired_normalized = paired_raw * inverse_rms;
    query_output[params.query_output_offset + head * HEAD_DIM + lane] =
      rotate_local(normalized, paired_normalized, lane);
    return;
  }

  let kv_head = head - QUERY_HEADS;
  let key_base = params.projected_offset + K_OFFSET + kv_head * HEAD_DIM;
  let raw_key = projected[key_base + lane];
  let sum = reduce_squared_sum(lane, raw_key);
  let inverse_rms = inverseSqrt(sum / f32(HEAD_DIM) + RMS_EPSILON);
  let paired_key = projected[key_base + (lane ^ 1u)] * inverse_rms;
  let normalized_key = raw_key * inverse_rms;

  let cache_index = cache_slot * KV_SIZE + kv_head * HEAD_DIM + lane;
  key_cache[params.key_cache_offset + cache_index] =
    rotate_local(normalized_key, paired_key, lane);

  let raw_value = projected[
    params.projected_offset + V_OFFSET + kv_head * HEAD_DIM + lane
  ];
  var value = raw_value;
  if (params.has_value_embedding != 0u) {
    let gate_logit = projected[
      params.projected_offset + VALUE_GATE_OFFSET + kv_head
    ];
    let gate = 2.0f * sigmoid(gate_logit);
    value += gate * value_embedding[
      params.value_embedding_offset + kv_head * HEAD_DIM + lane
    ];
  }
  value_cache[params.value_cache_offset + cache_index] = value;
}
`;

export const ATTENTION_PARTITION_KEYS = 32;
export const ATTENTION_PARTIAL_STRIDE = LIMITE_HEAD_DIM + 2;
export const ATTENTION_WORKGROUP_SIZE = LIMITE_HEAD_DIM;

/**
 * Split-K, single-token grouped-query attention and final Violetto gating.
 *
 * One module supplies two pipelines. Both use the same seven storage bindings,
 * below WebGPU's minimum limit of eight storage buffers per shader stage:
 *
 *   0: normalized/rotated query [10][128], f32
 *   1: layer key cache [capacity][2][128], f32
 *   2: layer value cache [capacity][2][128], f32
 *   3: partition scratch, read_write f32
 *   4: projected q/k/v/gates, f32 (attention gate starts at element 1792)
 *   5: pre-tanh XSA alpha [10], f32
 *   6: final attention output [10][128], f32
 *   7: AttentionParams uniform (64 bytes)
 *
 * Each partition record is 130 floats: local max, local softmax denominator,
 * then 128 unnormalized output components. `partial_partition_stride` is the
 * number of records reserved per query head and must be >= partition_count.
 *
 * AttentionParams as u32 values:
 *   [position, slot, valid_length, is_global,
 *    partition_count, partial_partition_stride, query_offset, key_cache_offset,
 *    value_cache_offset, projected_offset, xsa_offset, output_offset,
 *    partial_offset, cache_capacity, 0, 0]
 *
 * Encode `attention_partition` first with dispatch
 * `(10, partition_count, 1)`, then `attention_finalize` with dispatch
 * `(10, 1, 1)`. `partition_count` is ceil(key_count / 32). For a full local
 * window key_count is 1025, so partition_count is 33. Global layers retain the
 * full prefix and therefore need scratch sized for their maximum context.
 */
export const GQA_ATTENTION_WGSL = /* wgsl */ `
const QUERY_HEADS = ${LIMITE_QUERY_HEADS}u;
const KV_HEADS = ${LIMITE_KV_HEADS}u;
const GROUP_SIZE = ${LIMITE_GROUP_SIZE}u;
const HEAD_DIM = ${LIMITE_HEAD_DIM}u;
const KV_SIZE = ${LIMITE_KV_HEADS * LIMITE_HEAD_DIM}u;
const ATTENTION_GATE_OFFSET = ${
  LIMITE_QUERY_HEADS * LIMITE_HEAD_DIM + 2 * LIMITE_KV_HEADS * LIMITE_HEAD_DIM
}u;
const PARTITION_KEYS = ${ATTENTION_PARTITION_KEYS}u;
const PARTIAL_STRIDE = ${ATTENTION_PARTIAL_STRIDE}u;
const BACKWARD_SPAN = ${LIMITE_LOCAL_BACKWARD_SPAN}u;
const ATTENTION_SCALE = ${LIMITE_ATTENTION_SCALE}f;
const XSA_EPSILON = ${LIMITE_XSA_EPSILON}f;
const NEGATIVE_INFINITY = -3.402823e38f;

struct AttentionParams {
  position: u32,
  slot: u32,
  valid_length: u32,
  is_global: u32,
  partition_count: u32,
  partial_partition_stride: u32,
  query_offset: u32,
  key_cache_offset: u32,
  value_cache_offset: u32,
  projected_offset: u32,
  xsa_offset: u32,
  output_offset: u32,
  partial_offset: u32,
  cache_capacity: u32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<f32>;
@group(0) @binding(2) var<storage, read> value_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<storage, read> projected: array<f32>;
@group(0) @binding(5) var<storage, read> xsa_alpha: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> params: AttentionParams;

var<workgroup> reduction: array<f32, ${LIMITE_HEAD_DIM}>;
var<workgroup> shared_score: f32;
var<workgroup> shared_maximum: f32;
var<workgroup> shared_denominator: f32;

fn sigmoid(value: f32) -> f32 {
  return 1.0f / (1.0f + exp(-value));
}

fn key_range_start() -> u32 {
  if (params.is_global != 0u || params.position <= BACKWARD_SPAN) {
    return 0u;
  }
  return params.position - BACKWARD_SPAN;
}

fn key_range_end() -> u32 {
  return min(params.valid_length, params.position + 1u);
}

fn cache_slot_for_position(position: u32) -> u32 {
  if (params.is_global != 0u) {
    return position;
  }
  return position % ${LIMITE_LOCAL_KEY_COUNT}u;
}

@compute @workgroup_size(${ATTENTION_WORKGROUP_SIZE})
fn attention_partition(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let query_head = group.x;
  let partition_index = group.y;
  if (query_head >= QUERY_HEADS || partition_index >= params.partition_count) {
    return;
  }

  let kv_head = query_head / GROUP_SIZE;
  let range_start = key_range_start();
  let range_end = key_range_end();
  let partition_start = range_start + partition_index * PARTITION_KEYS;
  let partition_end = min(range_end, partition_start + PARTITION_KEYS);
  let record = params.partial_offset
    + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;

  if (partition_start >= partition_end) {
    if (lane == 0u) {
      partials[record] = NEGATIVE_INFINITY;
      partials[record + 1u] = 0.0f;
    }
    partials[record + 2u + lane] = 0.0f;
    return;
  }

  var local_maximum = NEGATIVE_INFINITY;
  var local_denominator = 0.0f;
  var numerator = 0.0f;
  let query_value = query[
    params.query_offset + query_head * HEAD_DIM + lane
  ];

  var key_position = partition_start;
  while (key_position < partition_end) {
    let cache_slot = cache_slot_for_position(key_position);
    // A full local ring is exactly 1025 slots. Global buffers must be sized to
    // the maximum accepted context; this guard leaves an invalid slot masked.
    if (cache_slot >= params.cache_capacity) {
      key_position += 1u;
      continue;
    }
    let cache_index = cache_slot * KV_SIZE + kv_head * HEAD_DIM + lane;
    reduction[lane] = query_value * key_cache[
      params.key_cache_offset + cache_index
    ];
    workgroupBarrier();

    var stride = HEAD_DIM >> 1u;
    while (stride > 0u) {
      if (lane < stride) {
        reduction[lane] += reduction[lane + stride];
      }
      workgroupBarrier();
      stride >>= 1u;
    }
    if (lane == 0u) {
      shared_score = reduction[0] * ATTENTION_SCALE;
    }
    workgroupBarrier();

    let new_maximum = max(local_maximum, shared_score);
    let old_scale = select(
      exp(local_maximum - new_maximum),
      0.0f,
      local_denominator == 0.0f,
    );
    let new_scale = exp(shared_score - new_maximum);
    let cached_value = value_cache[
      params.value_cache_offset + cache_index
    ];
    numerator = numerator * old_scale + cached_value * new_scale;
    local_denominator = local_denominator * old_scale + new_scale;
    local_maximum = new_maximum;
    key_position += 1u;
  }

  if (lane == 0u) {
    partials[record] = local_maximum;
    partials[record + 1u] = local_denominator;
  }
  partials[record + 2u + lane] = numerator;
}

@compute @workgroup_size(${ATTENTION_WORKGROUP_SIZE})
fn attention_finalize(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let query_head = group.x;
  let current_slot = cache_slot_for_position(params.position);
  if (query_head >= QUERY_HEADS || current_slot >= params.cache_capacity) {
    return;
  }

  if (lane == 0u) {
    var maximum = NEGATIVE_INFINITY;
    for (var partition_index = 0u; partition_index < params.partition_count; partition_index += 1u) {
      let record = params.partial_offset
        + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
      maximum = max(maximum, partials[record]);
    }
    shared_maximum = maximum;

    var denominator = 0.0f;
    if (maximum != NEGATIVE_INFINITY) {
      for (var partition_index = 0u; partition_index < params.partition_count; partition_index += 1u) {
        let record = params.partial_offset
          + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
        let partition_maximum = partials[record];
        if (partition_maximum != NEGATIVE_INFINITY) {
          denominator += partials[record + 1u]
            * exp(partition_maximum - maximum);
        }
      }
    }
    shared_denominator = denominator;
  }
  workgroupBarrier();

  var attention_value = 0.0f;
  if (shared_denominator > 0.0f) {
    for (var partition_index = 0u; partition_index < params.partition_count; partition_index += 1u) {
      let record = params.partial_offset
        + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
      let partition_maximum = partials[record];
      if (partition_maximum != NEGATIVE_INFINITY) {
        attention_value += partials[record + 2u + lane]
          * exp(partition_maximum - shared_maximum);
      }
    }
    attention_value /= shared_denominator;
  }

  let kv_head = query_head / GROUP_SIZE;
  let current_value_index = current_slot * KV_SIZE + kv_head * HEAD_DIM + lane;
  let current_value = value_cache[
    params.value_cache_offset + current_value_index
  ];
  reduction[lane] = current_value * current_value;
  workgroupBarrier();
  var stride = HEAD_DIM >> 1u;
  while (stride > 0u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  let direction = current_value / max(sqrt(reduction[0]), XSA_EPSILON);
  workgroupBarrier();

  reduction[lane] = attention_value * direction;
  workgroupBarrier();
  stride = HEAD_DIM >> 1u;
  while (stride > 0u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  let alpha = xsa_alpha[params.xsa_offset + query_head];
  let xsa_value = attention_value - alpha * reduction[0] * direction;
  let gate_logit = projected[
    params.projected_offset + ATTENTION_GATE_OFFSET + query_head
  ];
  let gated = xsa_value * (2.0f * sigmoid(gate_logit));
  output[params.output_offset + query_head * HEAD_DIM + lane] = gated;
}
`;

export const ELEMENTWISE_WORKGROUP_SIZE = 256;

/**
 * SwiGLU activation over Limite's fused gate/up projection.
 *
 * Input layout is gate[3328], up[3328]. Bindings are input f32, output f32,
 * and a 32-byte uniform. Uniform u32 layout:
 *   [input_offset, output_offset, intermediate_size, 0, 0, 0, 0, 0]
 * Dispatch ceil(intermediate_size / 256), exactly (13, 1, 1) for Limite.
 */
export const SWIGLU_WGSL = /* wgsl */ `
struct SwiGluParams {
  input_offset: u32,
  output_offset: u32,
  intermediate_size: u32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
  _padding_3: u32,
  _padding_4: u32,
}

@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SwiGluParams;

@compute @workgroup_size(${ELEMENTWISE_WORKGROUP_SIZE})
fn swiglu(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= params.intermediate_size) {
    return;
  }
  let gate = gate_up[params.input_offset + index];
  let up = gate_up[
    params.input_offset + params.intermediate_size + index
  ];
  let silu = gate / (1.0f + exp(-gate));
  output[params.output_offset + index] = silu * up;
}
`;

export const SAMPLER_WORKGROUP_SIZE = 256;
export const SAMPLER_TOP_K = 50;
export const SAMPLER_CANDIDATES_PER_PARTITION = SAMPLER_TOP_K;

/**
 * Fully GPU-resident top-50 / nucleus sampler.
 *
 * `sample_partitions` bitonic-sorts each 256-logit partition and writes its
 * best 50 raw logits. `sample_top_50` performs a parallel k-way merge of those
 * already-sorted partition lists, applies the monotonic sigmoid softcap,
 * temperature, and top-p, advances a persistent xorshift32 state, and writes
 * the chosen id directly to the next-token buffer.
 * No logits or random values cross to the CPU.
 *
 * Bindings (five storage buffers, comfortably below the WebGPU minimum):
 *   0: raw f32 logits
 *   1: candidate values, read_write f32
 *   2: candidate token ids, read_write u32
 *   3: next-token ids, read_write u32
 *   4: persistent RNG state, read_write u32
 *   5: SamplerParams uniform (48 bytes)
 *
 * Params layout:
 *   u32 vocab_size, logits_offset, candidate_offset, candidate_count;
 *   u32 token_output_offset, rng_offset, partition_count, _padding_0;
 *   f32 temperature, top_p; u32 _padding_1, _padding_2.
 *
 * Dispatch `sample_partitions` with `(ceil(vocab_size / 256), 1, 1)`, then
 * `sample_top_50` with `(1, 1, 1)`. For Limite these are 593 and 1. Candidate
 * buffers need `partition_count * 50` entries (29650 for the full vocabulary).
 */
export const GPU_TOP_K_TOP_P_SAMPLER_WGSL = /* wgsl */ `
const WORKGROUP_SIZE = ${SAMPLER_WORKGROUP_SIZE}u;
const TOP_K = ${SAMPLER_TOP_K}u;
const NEGATIVE_INFINITY = -3.402823e38f;

struct SamplerParams {
  vocab_size: u32,
  logits_offset: u32,
  candidate_offset: u32,
  candidate_count: u32,
  token_output_offset: u32,
  rng_offset: u32,
  partition_count: u32,
  _padding_0: u32,
  temperature: f32,
  top_p: f32,
  _padding_1: u32,
  _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> candidate_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> candidate_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> next_token: array<u32>;
@group(0) @binding(4) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(5) var<uniform> params: SamplerParams;

var<workgroup> sort_values: array<f32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> sort_ids: array<u32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> reduce_values: array<f32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> reduce_ids: array<u32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> reduce_partitions: array<u32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> partition_cursors: array<u32, ${Math.ceil(LIMITE_VOCAB_SIZE / SAMPLER_WORKGROUP_SIZE)}>;
var<workgroup> top_values: array<f32, ${SAMPLER_TOP_K}>;
var<workgroup> top_ids: array<u32, ${SAMPLER_TOP_K}>;
var<workgroup> probabilities: array<f32, ${SAMPLER_TOP_K}>;
var<workgroup> valid_top_count: u32;

fn better(value_a: f32, id_a: u32, value_b: f32, id_b: u32) -> bool {
  return value_a > value_b || (value_a == value_b && id_a < id_b);
}

fn stable_sigmoid(value: f32) -> f32 {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + exp(-value));
  }
  let exponential = exp(value);
  return exponential / (1.0f + exponential);
}

fn softcap(value: f32) -> f32 {
  return 23.0f * stable_sigmoid((value + 5.0f) / 7.5f);
}

fn random_unit() -> f32 {
  var state = rng_state[params.rng_offset];
  if (state == 0u) {
    state = 0x6d2b79f5u;
  }
  state ^= state << 13u;
  state ^= state >> 17u;
  state ^= state << 5u;
  rng_state[params.rng_offset] = state;
  return f32(state >> 8u) * (1.0f / 16777216.0f);
}

@compute @workgroup_size(${SAMPLER_WORKGROUP_SIZE})
fn sample_partitions(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let partition_index = group.x;
  if (partition_index >= params.partition_count) {
    return;
  }
  let token = partition_index * WORKGROUP_SIZE + lane;
  var value = NEGATIVE_INFINITY;
  var id = 0xffffffffu;
  if (token < params.vocab_size) {
    let candidate = logits[params.logits_offset + token];
    // Both comparisons are false for NaN; either rejects the corresponding
    // infinity while accepting every finite f32 value.
    if (candidate >= NEGATIVE_INFINITY && candidate <= -NEGATIVE_INFINITY) {
      value = candidate;
      id = token;
    }
  }
  sort_values[lane] = value;
  sort_ids[lane] = id;
  workgroupBarrier();

  // Ascending bitonic order. The best entries finish at the high end.
  var width = 2u;
  while (width <= WORKGROUP_SIZE) {
    var distance = width >> 1u;
    while (distance > 0u) {
      let other = lane ^ distance;
      if (other > lane) {
        let value_a = sort_values[lane];
        let id_a = sort_ids[lane];
        let value_b = sort_values[other];
        let id_b = sort_ids[other];
        let ascending = (lane & width) == 0u;
        let should_swap = select(
          better(value_b, id_b, value_a, id_a),
          better(value_a, id_a, value_b, id_b),
          ascending,
        );
        if (should_swap) {
          sort_values[lane] = value_b;
          sort_ids[lane] = id_b;
          sort_values[other] = value_a;
          sort_ids[other] = id_a;
        }
      }
      workgroupBarrier();
      distance >>= 1u;
    }
    width <<= 1u;
  }

  if (lane < TOP_K) {
    let source = WORKGROUP_SIZE - 1u - lane;
    let destination = params.candidate_offset + partition_index * TOP_K + lane;
    candidate_values[destination] = sort_values[source];
    candidate_ids[destination] = sort_ids[source];
  }
}

@compute @workgroup_size(${SAMPLER_WORKGROUP_SIZE})
fn sample_top_50(@builtin(local_invocation_index) lane: u32) {
  var partition_index = lane;
  while (partition_index < params.partition_count) {
    partition_cursors[partition_index] = 0u;
    partition_index += WORKGROUP_SIZE;
  }
  if (lane == 0u) {
    valid_top_count = 0u;
  }
  workgroupBarrier();

  // Each partition list is already sorted best-first. Selecting only its
  // current head makes this an exact k-way merge and avoids rescanning all
  // 29,650 candidates for every one of the final 50 ranks.
  for (var rank = 0u; rank < TOP_K; rank += 1u) {
    var lane_value = NEGATIVE_INFINITY;
    var lane_id = 0xffffffffu;
    var lane_partition = 0xffffffffu;
    partition_index = lane;
    while (partition_index < params.partition_count) {
      let cursor = partition_cursors[partition_index];
      let absolute = params.candidate_offset + partition_index * TOP_K + cursor;
      let value = candidate_values[absolute];
      let id = candidate_ids[absolute];
      if (better(value, id, lane_value, lane_id)) {
        lane_value = value;
        lane_id = id;
        lane_partition = partition_index;
      }
      partition_index += WORKGROUP_SIZE;
    }
    reduce_values[lane] = lane_value;
    reduce_ids[lane] = lane_id;
    reduce_partitions[lane] = lane_partition;
    workgroupBarrier();

    var stride = WORKGROUP_SIZE >> 1u;
    while (stride > 0u) {
      if (lane < stride) {
        let other_value = reduce_values[lane + stride];
        let other_id = reduce_ids[lane + stride];
        if (better(other_value, other_id, reduce_values[lane], reduce_ids[lane])) {
          reduce_values[lane] = other_value;
          reduce_ids[lane] = other_id;
          reduce_partitions[lane] = reduce_partitions[lane + stride];
        }
      }
      workgroupBarrier();
      stride >>= 1u;
    }

    if (lane == 0u) {
      top_values[rank] = reduce_values[0];
      top_ids[rank] = reduce_ids[0];
      let selected_partition = reduce_partitions[0];
      if (reduce_ids[0] != 0xffffffffu && selected_partition != 0xffffffffu) {
        valid_top_count = rank + 1u;
        partition_cursors[selected_partition] += 1u;
      }
    }
    workgroupBarrier();
  }

  if (lane != 0u) {
    return;
  }
  if (valid_top_count == 0u) {
    next_token[params.token_output_offset] = 151643u;
    return;
  }
  if (params.temperature <= 0.0f) {
    next_token[params.token_output_offset] = top_ids[0];
    return;
  }

  let maximum = softcap(top_values[0]);
  var total = 0.0f;
  for (var rank = 0u; rank < valid_top_count; rank += 1u) {
    let probability = exp((softcap(top_values[rank]) - maximum) / params.temperature);
    probabilities[rank] = probability;
    total += probability;
  }

  let target_p = clamp(params.top_p, 0.0f, 1.0f);
  var kept_total = 0.0f;
  var kept = 0u;
  while (kept < valid_top_count) {
    kept_total += probabilities[kept];
    kept += 1u;
    if (kept_total / total >= target_p) {
      break;
    }
  }

  var pick = random_unit() * kept_total;
  for (var rank = 0u; rank < kept; rank += 1u) {
    pick -= probabilities[rank];
    if (pick <= 0.0f) {
      next_token[params.token_output_offset] = top_ids[rank];
      return;
    }
  }
  next_token[params.token_output_offset] = top_ids[kept - 1u];
}
`;
