/**
 * Fixed-shape WebGPU decode operators for Limite 1B Violetto.
 *
 * These modules intentionally expose plain WGSL rather than constructing
 * pipelines. The engine owns every mutable buffer and can therefore create
 * pipelines/bind groups once, then encode the same command sequence for every
 * generated token. All element offsets below are scalar offsets, not bytes.
 *
 * Every model matrix preserves the checkpoint's exact BF16 words, packed two
 * per u32. Matrices are stored row-major in immutable artifact shards.
 */

import { BF16_ROUND_WGSL } from "./shaders";

/** Every possible BF16 bit pattern, indexed by its upper 16 f32 bits. */
export const BF16_LOOKUP_ENTRY_COUNT = 1 << 16;

export const LIMITE_HIDDEN_SIZE = 1_280;
export const LIMITE_INTERMEDIATE_SIZE = 3_328;
export const LIMITE_QUERY_HEADS = 10;
export const LIMITE_KV_HEADS = 2;
export const LIMITE_HEAD_DIM = 128;
export const LIMITE_GROUP_SIZE = 5;
export const LIMITE_PADDED_VOCAB_SIZE = 151_680;
export const LIMITE_TOKENIZER_VOCAB_SIZE = 151_667;
export const LIMITE_LOCAL_BACKWARD_SPAN = 1_024;
export const LIMITE_LOCAL_KEY_COUNT = LIMITE_LOCAL_BACKWARD_SPAN + 1;
export const LIMITE_ATTENTION_SCALE = 0.1;
export const LIMITE_RMS_EPSILON = 1 / 128;
export const LIMITE_XSA_EPSILON = 1e-4;

export const PACKED_BF16_EMBED_WORKGROUP_SIZE = 256;

/** Exact packed-BF16 gather used by the tied and value embeddings. */
export const PACKED_BF16_EMBEDDING_WGSL = /* wgsl */ `
struct PackedBf16EmbeddingParams {
  token_offset: u32,
  output_offset: u32,
  row_width: u32,
  row_count: u32,
  row_start: u32,
  clear_on_miss: u32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: PackedBf16EmbeddingParams;

fn unpack_bf16(word: u32, high: bool) -> f32 {
  let bits = select(word << 16u, word & 0xffff0000u, high);
  return bitcast<f32>(bits);
}

@compute @workgroup_size(${PACKED_BF16_EMBED_WORKGROUP_SIZE})
fn packed_bf16_embedding(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let column = invocation.x;
  if (column >= params.row_width) {
    return;
  }

  let token = token_ids[params.token_offset];
  if (token < params.row_start || token >= params.row_start + params.row_count) {
    if (params.clear_on_miss != 0u) {
      output[params.output_offset + column] = 0.0f;
    }
    return;
  }

  let local_row = token - params.row_start;
  let element = local_row * params.row_width + column;
  let word = matrix[element >> 1u];
  output[params.output_offset + column] = unpack_bf16(word, (element & 1u) != 0u);
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
 *   3: layer key cache, packed BF16 (two values per u32)
 *   4: layer value cache, packed BF16 (two values per u32)
 *   5: QkvPostprocessParams uniform (48 bytes)
 *   6: precomputed local RoPE cosine/signed-sine pairs, f32
 *
 * Params as u32 values:
 *   [position, slot, cache_capacity, is_global,
 *    has_value_embedding, projected_offset, value_embedding_offset,
 *    query_output_offset, key_cache_offset, value_cache_offset, 0, 0]
 * Cache offsets count packed u32 words; activation offsets count f32 elements.
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
  sigmoid_offset: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> value_embedding: array<f32>;
@group(0) @binding(2) var<storage, read_write> query_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> key_cache: array<u32>;
@group(0) @binding(4) var<storage, read_write> value_cache: array<u32>;
@group(0) @binding(5) var<uniform> params: QkvPostprocessParams;
@group(0) @binding(6) var<storage, read> rope_factors: array<f32>;
@group(0) @binding(7) var<storage, read> nonlinearities: array<f32>;

${BF16_ROUND_WGSL}

var<workgroup> squared_sum: array<f32, ${LIMITE_HEAD_DIM}>;
var<workgroup> cache_keys: array<f32, ${LIMITE_HEAD_DIM}>;
var<workgroup> cache_values: array<f32, ${LIMITE_HEAD_DIM}>;

fn exact_bf16_sigmoid(value: f32) -> f32 {
  return nonlinearities[
    params.sigmoid_offset + (bitcast<u32>(value) >> 16u)
  ];
}

fn rotate_local(value: f32, paired_value: f32, dimension: u32) -> f32 {
  if (params.is_global != 0u || dimension >= 64u) {
    return value;
  }
  let factor = (params.position * 64u + dimension) * 2u;
  // The official BF16 graph materializes both products before the BF16 add.
  let direct = to_bf16(value * rope_factors[factor]);
  let paired = to_bf16(paired_value * rope_factors[factor + 1u]);
  return to_bf16(direct + paired);
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

fn pack_bf16_pair(low: f32, high: f32) -> u32 {
  return (bitcast<u32>(low) >> 16u) | (bitcast<u32>(high) & 0xffff0000u);
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
    let normalized = to_bf16(raw * inverse_rms);
    let paired_normalized = to_bf16(paired_raw * inverse_rms);
    query_output[params.query_output_offset + head * HEAD_DIM + lane] =
      rotate_local(normalized, paired_normalized, lane);
    return;
  }

  let kv_head = head - QUERY_HEADS;
  let key_base = params.projected_offset + K_OFFSET + kv_head * HEAD_DIM;
  let raw_key = projected[key_base + lane];
  let sum = reduce_squared_sum(lane, raw_key);
  let inverse_rms = inverseSqrt(sum / f32(HEAD_DIM) + RMS_EPSILON);
  let paired_key = to_bf16(
    projected[key_base + (lane ^ 1u)] * inverse_rms,
  );
  let normalized_key = to_bf16(raw_key * inverse_rms);

  cache_keys[lane] = rotate_local(normalized_key, paired_key, lane);

  let raw_value = projected[
    params.projected_offset + V_OFFSET + kv_head * HEAD_DIM + lane
  ];
  var value = raw_value;
  if (params.has_value_embedding != 0u) {
    let gate_logit = projected[
      params.projected_offset + VALUE_GATE_OFFSET + kv_head
    ];
    // Sigmoid, scaling, product, and residual add are distinct BF16 ops upstream.
    let sigmoid_gate = exact_bf16_sigmoid(gate_logit);
    let gate = to_bf16(2.0f * sigmoid_gate);
    let value_delta = to_bf16(gate * value_embedding[
      params.value_embedding_offset + kv_head * HEAD_DIM + lane
    ]);
    value = to_bf16(raw_value + value_delta);
  }
  cache_values[lane] = to_bf16(value);
  workgroupBarrier();

  // One lane owns each complete packed word. This avoids read-modify-write
  // races and never carries stale half-words across local-ring overwrites.
  if (lane < HEAD_DIM / 2u) {
    let element = lane * 2u;
    let cache_word = (
      cache_slot * KV_SIZE + kv_head * HEAD_DIM
    ) / 2u + lane;
    key_cache[params.key_cache_offset + cache_word] = pack_bf16_pair(
      cache_keys[element],
      cache_keys[element + 1u],
    );
    value_cache[params.value_cache_offset + cache_word] = pack_bf16_pair(
      cache_values[element],
      cache_values[element + 1u],
    );
  }
}
`;

export const ATTENTION_PARTITION_KEYS = 32;
export const ATTENTION_PARTIAL_STRIDE = LIMITE_HEAD_DIM + 2;
export const ATTENTION_WORKGROUP_SIZE = 192;
const ATTENTION_HEAD_LANES = 32;
const ATTENTION_COMPONENTS_PER_LANE = LIMITE_HEAD_DIM / ATTENTION_HEAD_LANES;
const ATTENTION_FINALIZE_SCALE_TILE = 256;

/**
 * Split-K, single-token grouped-query attention and final Violetto gating.
 *
 * One module supplies two pipelines. Both use the same seven storage bindings,
 * below WebGPU's minimum limit of eight storage buffers per shader stage:
 *
 *   0: normalized/rotated query [10][128], f32
 *   1: layer key cache [capacity][2][128], packed BF16
 *   2: layer value cache [capacity][2][128], packed BF16
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
 * Cache offsets count packed u32 words; all other offsets count f32 elements.
 *
 * Each partition workgroup fuses the five query heads that share one KV head.
 * Its first 160 lanes initially map one lane to each (query head, key) score.
 * Once all scores are available, those lanes become five 32-lane head cohorts
 * with four output dimensions per lane. Keys are staged transposed for
 * contiguous score reads, then the same workgroup tile is reused row-major for
 * values. Three uniform barriers publish the key tile, scores, and value/scale
 * tile respectively.
 *
 * Encode `attention_partition` first with dispatch
 * `(2, partition_count, 1)`, then `attention_finalize` with dispatch
 * `(2, 1, 1)`. `partition_count` is ceil(key_count / 32). For a full local
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
const HEAD_LANES = ${ATTENTION_HEAD_LANES}u;
const COMPONENTS_PER_LANE = ${ATTENTION_COMPONENTS_PER_LANE}u;
const COMPUTE_LANES = ${LIMITE_GROUP_SIZE * ATTENTION_HEAD_LANES}u;
const CACHE_WORDS_PER_KEY = HEAD_DIM / 2u;
const PARTITION_WORDS = PARTITION_KEYS * CACHE_WORDS_PER_KEY;
const PARTITION_SCORE_COUNT = GROUP_SIZE * PARTITION_KEYS;
const PARTITION_SCORE_OFFSET = 0u;
const PARTITION_OLD_SCALE_OFFSET = PARTITION_SCORE_COUNT;
const PARTITION_NEW_SCALE_OFFSET = 2u * PARTITION_SCORE_COUNT;
const FINALIZE_SCALE_TILE = ${ATTENTION_FINALIZE_SCALE_TILE}u;

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
  sigmoid_offset: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key_cache: array<u32>;
@group(0) @binding(2) var<storage, read> value_cache: array<u32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;
@group(0) @binding(4) var<storage, read> projected: array<f32>;
@group(0) @binding(5) var<storage, read> xsa_alpha: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<uniform> params: AttentionParams;

${BF16_ROUND_WGSL}

var<workgroup> reduction: array<f32, ${LIMITE_GROUP_SIZE * LIMITE_HEAD_DIM}>;
var<workgroup> cached_partition_words: array<u32, ${
  ATTENTION_PARTITION_KEYS * (LIMITE_HEAD_DIM / 2)
}>;
var<workgroup> cached_value_words: array<u32, ${LIMITE_HEAD_DIM / 2}>;
var<workgroup> shared_maximum: array<f32, ${LIMITE_GROUP_SIZE}>;
var<workgroup> shared_denominator: array<f32, ${LIMITE_GROUP_SIZE}>;
// The finalizer uses this as five 256-partition scale tiles. The partition
// pipeline reuses its first 480 values for score, old-scale, and new-scale
// tables. Together with the 8 KiB cache tile, all module workgroup storage is
// 16,168 bytes, below WebGPU's guaranteed 16 KiB limit.
var<workgroup> partition_scales: array<f32, ${
  LIMITE_GROUP_SIZE * ATTENTION_FINALIZE_SCALE_TILE
}>;

fn exact_bf16_sigmoid(value: f32) -> f32 {
  return xsa_alpha[
    params.sigmoid_offset + (bitcast<u32>(value) >> 16u)
  ];
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

fn unpack_bf16_low(word: u32) -> f32 {
  return bitcast<f32>((word & 0xffffu) << 16u);
}

fn unpack_bf16_high(word: u32) -> f32 {
  return bitcast<f32>(word & 0xffff0000u);
}

fn unpack_bf16_quad(first: u32, second: u32) -> vec4<f32> {
  return vec4<f32>(
    unpack_bf16_low(first),
    unpack_bf16_high(first),
    unpack_bf16_low(second),
    unpack_bf16_high(second),
  );
}

fn unpack_bf16_component(word: u32, component: u32) -> f32 {
  return select(
    unpack_bf16_low(word),
    unpack_bf16_high(word),
    (component & 1u) != 0u,
  );
}

// The original partition reduction stores all 128 products, then reduces at
// strides 64, 32, ..., 1. Visiting dimensions in bit-reversed order lets one
// score lane reproduce that exact f32 addition tree with a seven-value binary
// carry stack instead of 128 private values or seven workgroup barriers.
fn partition_score(relative_head: u32, key_in_partition: u32) -> f32 {
  var tree: array<f32, 7>;
  var dot = 0.0f;
  for (var leaf = 0u; leaf < HEAD_DIM; leaf += 1u) {
    let component = reverseBits(leaf) >> 25u;
    let packed = cached_partition_words[
      (component >> 1u) * PARTITION_KEYS + key_in_partition
    ];
    var subtotal = reduction[relative_head * HEAD_DIM + component]
      * unpack_bf16_component(packed, component);
    var branch = leaf;
    var level = 0u;
    loop {
      if ((branch & 1u) == 0u) {
        tree[level] = subtotal;
        break;
      }
      subtotal = tree[level] + subtotal;
      level += 1u;
      branch >>= 1u;
      if (level == 7u) {
        dot = subtotal;
        break;
      }
    }
  }
  return dot * ATTENTION_SCALE;
}

fn reduce_head_planes(relative_head: u32, head_lane: u32) {
  workgroupBarrier();
  var stride = HEAD_DIM >> 1u;
  while (stride > 0u) {
    if (relative_head < GROUP_SIZE) {
      var component = head_lane;
      while (component < stride) {
        let index = relative_head * HEAD_DIM + component;
        reduction[index] += reduction[index + stride];
        component += HEAD_LANES;
      }
    }
    workgroupBarrier();
    stride >>= 1u;
  }
}

fn reduce_first_plane(lane: u32) {
  workgroupBarrier();
  var stride = HEAD_DIM >> 1u;
  while (stride > 0u) {
    if (lane < HEAD_LANES) {
      var component = lane;
      while (component < stride) {
        reduction[component] += reduction[component + stride];
        component += HEAD_LANES;
      }
    }
    workgroupBarrier();
    stride >>= 1u;
  }
}

@compute @workgroup_size(${ATTENTION_WORKGROUP_SIZE})
fn attention_partition(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let kv_head = group.x;
  let partition_index = group.y;
  if (kv_head >= KV_HEADS || partition_index >= params.partition_count) {
    return;
  }

  let relative_head = lane / HEAD_LANES;
  let head_lane = lane % HEAD_LANES;
  let is_active = relative_head < GROUP_SIZE;
  let query_head = kv_head * GROUP_SIZE + relative_head;
  let component_base = head_lane * COMPONENTS_PER_LANE;
  let range_start = key_range_start();
  let range_end = key_range_end();
  let partition_start = range_start + partition_index * PARTITION_KEYS;
  let partition_end = min(range_end, partition_start + PARTITION_KEYS);
  let record = params.partial_offset
    + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;

  if (partition_start >= partition_end) {
    if (is_active) {
      if (head_lane == 0u) {
        partials[record] = NEGATIVE_INFINITY;
        partials[record + 1u] = 0.0f;
      }
      partials[record + 2u + component_base] = 0.0f;
      partials[record + 3u + component_base] = 0.0f;
      partials[record + 4u + component_base] = 0.0f;
      partials[record + 5u + component_base] = 0.0f;
    }
    return;
  }

  let partition_key_count = partition_end - partition_start;

  // Stage the five queries once. The existing reduction plane is idle until
  // the finalizer and is exactly the required 5 * 128 f32 values.
  let query_group_base = params.query_offset + kv_head * GROUP_SIZE * HEAD_DIM;
  var staged_query = lane;
  while (staged_query < GROUP_SIZE * HEAD_DIM) {
    reduction[staged_query] = query[query_group_base + staged_query];
    staged_query += ${ATTENTION_WORKGROUP_SIZE}u;
  }

  // Keys use a word-major transpose. For a fixed component, each 32-lane head
  // cohort therefore reads the 32 partition keys contiguously. The loader
  // traverses the cache row-major for coalesced source reads and transposes
  // only its workgroup-memory destination.
  var tile_word = lane;
  while (tile_word < PARTITION_WORDS) {
    let key_in_partition = tile_word / CACHE_WORDS_PER_KEY;
    let word_in_key = tile_word % CACHE_WORDS_PER_KEY;
    var packed = 0u;
    if (key_in_partition < partition_key_count) {
      let key_position = partition_start + key_in_partition;
      let cache_slot = cache_slot_for_position(key_position);
      if (cache_slot < params.cache_capacity) {
        let cache_word_base = (
          cache_slot * KV_SIZE + kv_head * HEAD_DIM
        ) >> 1u;
        packed = key_cache[
          params.key_cache_offset + cache_word_base + word_in_key
        ];
      }
    }
    cached_partition_words[
      word_in_key * PARTITION_KEYS + key_in_partition
    ] = packed;
    tile_word += ${ATTENTION_WORKGROUP_SIZE}u;
  }

  if (is_active && head_lane == 0u) {
    shared_maximum[relative_head] = NEGATIVE_INFINITY;
    shared_denominator[relative_head] = 0.0f;
  }
  workgroupBarrier();

  // The first 160 lanes are now one lane per (query head, key) score.
  if (is_active && head_lane < partition_key_count) {
    let score_index = relative_head * PARTITION_KEYS + head_lane;
    let key_position = partition_start + head_lane;
    let cache_slot = cache_slot_for_position(key_position);
    var score = NEGATIVE_INFINITY;
    if (cache_slot < params.cache_capacity) {
      score = partition_score(relative_head, head_lane);
    }
    partition_scales[PARTITION_SCORE_OFFSET + score_index] = score;
  }
  workgroupBarrier();

  // All score readers have left the key tile. Overwrite it with row-major
  // values, which makes each four-component output lane's pair of words
  // contiguous. At the same time, one leader per head walks scores in key
  // order and records the exact online-softmax scales for the value pass.
  tile_word = lane;
  while (tile_word < PARTITION_WORDS) {
    let key_in_partition = tile_word / CACHE_WORDS_PER_KEY;
    let word_in_key = tile_word % CACHE_WORDS_PER_KEY;
    var packed = 0u;
    if (key_in_partition < partition_key_count) {
      let key_position = partition_start + key_in_partition;
      let cache_slot = cache_slot_for_position(key_position);
      if (cache_slot < params.cache_capacity) {
        let cache_word_base = (
          cache_slot * KV_SIZE + kv_head * HEAD_DIM
        ) >> 1u;
        packed = value_cache[
          params.value_cache_offset + cache_word_base + word_in_key
        ];
      }
    }
    cached_partition_words[tile_word] = packed;
    tile_word += ${ATTENTION_WORKGROUP_SIZE}u;
  }

  if (is_active && head_lane == 0u) {
    var maximum = shared_maximum[relative_head];
    var denominator = shared_denominator[relative_head];
    for (var key = 0u; key < partition_key_count; key += 1u) {
      let score_index = relative_head * PARTITION_KEYS + key;
      let key_position = partition_start + key;
      let cache_slot = cache_slot_for_position(key_position);
      var old_scale = 1.0f;
      var new_scale = 0.0f;
      if (cache_slot < params.cache_capacity) {
        let score = partition_scales[PARTITION_SCORE_OFFSET + score_index];
        let score_raises_maximum = score > maximum;
        let new_maximum = select(maximum, score, score_raises_maximum);
        new_scale = 1.0f;
        if (score_raises_maximum) {
          old_scale = exp(maximum - score);
        } else {
          new_scale = exp(score - maximum);
        }
        old_scale = select(old_scale, 0.0f, denominator == 0.0f);
        denominator = denominator * old_scale + new_scale;
        maximum = new_maximum;
      }
      partition_scales[PARTITION_OLD_SCALE_OFFSET + score_index] = old_scale;
      partition_scales[PARTITION_NEW_SCALE_OFFSET + score_index] = new_scale;
    }
    shared_maximum[relative_head] = maximum;
    shared_denominator[relative_head] = denominator;
  }
  workgroupBarrier();

  var numerator_0 = 0.0f;
  var numerator_1 = 0.0f;
  var numerator_2 = 0.0f;
  var numerator_3 = 0.0f;
  if (is_active) {
    for (var key = 0u; key < partition_key_count; key += 1u) {
      let score_index = relative_head * PARTITION_KEYS + key;
      let first_word = key * CACHE_WORDS_PER_KEY + head_lane * 2u;
      let cached_value = unpack_bf16_quad(
        cached_partition_words[first_word],
        cached_partition_words[first_word + 1u],
      );
      let old_scale = partition_scales[
        PARTITION_OLD_SCALE_OFFSET + score_index
      ];
      let new_scale = partition_scales[
        PARTITION_NEW_SCALE_OFFSET + score_index
      ];
      numerator_0 = numerator_0 * old_scale + cached_value.x * new_scale;
      numerator_1 = numerator_1 * old_scale + cached_value.y * new_scale;
      numerator_2 = numerator_2 * old_scale + cached_value.z * new_scale;
      numerator_3 = numerator_3 * old_scale + cached_value.w * new_scale;
    }
  }

  if (is_active) {
    if (head_lane == 0u) {
      partials[record] = shared_maximum[relative_head];
      partials[record + 1u] = shared_denominator[relative_head];
    }
    partials[record + 2u + component_base] = numerator_0;
    partials[record + 3u + component_base] = numerator_1;
    partials[record + 4u + component_base] = numerator_2;
    partials[record + 5u + component_base] = numerator_3;
  }
}

@compute @workgroup_size(${ATTENTION_WORKGROUP_SIZE})
fn attention_finalize(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let kv_head = group.x;
  let current_slot = cache_slot_for_position(params.position);
  if (kv_head >= KV_HEADS || current_slot >= params.cache_capacity) {
    return;
  }

  let relative_head = lane / HEAD_LANES;
  let head_lane = lane % HEAD_LANES;
  let is_active = relative_head < GROUP_SIZE;
  let query_head = kv_head * GROUP_SIZE + relative_head;
  let component_base = head_lane * COMPONENTS_PER_LANE;

  if (is_active && head_lane == 0u) {
    var maximum = NEGATIVE_INFINITY;
    for (var partition_index = 0u; partition_index < params.partition_count; partition_index += 1u) {
      let record = params.partial_offset
        + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
      maximum = max(maximum, partials[record]);
    }
    shared_maximum[relative_head] = maximum;
    shared_denominator[relative_head] = 0.0f;
  }
  workgroupBarrier();

  var attention_0 = 0.0f;
  var attention_1 = 0.0f;
  var attention_2 = 0.0f;
  var attention_3 = 0.0f;
  var tile_start = 0u;
  while (tile_start < params.partition_count) {
    let tile_count = min(
      FINALIZE_SCALE_TILE,
      params.partition_count - tile_start,
    );
    if (is_active && head_lane == 0u) {
      let scale_base = relative_head * FINALIZE_SCALE_TILE;
      var denominator = shared_denominator[relative_head];
      for (var tile_index = 0u; tile_index < tile_count; tile_index += 1u) {
        let partition_index = tile_start + tile_index;
        let record = params.partial_offset
          + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
        let partition_maximum = partials[record];
        var scale = 0.0f;
        if (
          shared_maximum[relative_head] != NEGATIVE_INFINITY
          && partition_maximum != NEGATIVE_INFINITY
        ) {
          scale = exp(partition_maximum - shared_maximum[relative_head]);
          denominator += partials[record + 1u] * scale;
        }
        partition_scales[scale_base + tile_index] = scale;
      }
      shared_denominator[relative_head] = denominator;
    }
    workgroupBarrier();

    if (is_active) {
      let scale_base = relative_head * FINALIZE_SCALE_TILE;
      for (var tile_index = 0u; tile_index < tile_count; tile_index += 1u) {
        let partition_index = tile_start + tile_index;
        let record = params.partial_offset
          + (query_head * params.partial_partition_stride + partition_index) * PARTIAL_STRIDE;
        let scale = partition_scales[scale_base + tile_index];
        attention_0 += partials[record + 2u + component_base] * scale;
        attention_1 += partials[record + 3u + component_base] * scale;
        attention_2 += partials[record + 4u + component_base] * scale;
        attention_3 += partials[record + 5u + component_base] * scale;
      }
    }
    workgroupBarrier();
    tile_start += FINALIZE_SCALE_TILE;
  }

  var bf16_attention_0 = 0.0f;
  var bf16_attention_1 = 0.0f;
  var bf16_attention_2 = 0.0f;
  var bf16_attention_3 = 0.0f;
  if (is_active && shared_denominator[relative_head] > 0.0f) {
    let denominator = shared_denominator[relative_head];
    bf16_attention_0 = to_bf16(attention_0 / denominator);
    bf16_attention_1 = to_bf16(attention_1 / denominator);
    bf16_attention_2 = to_bf16(attention_2 / denominator);
    bf16_attention_3 = to_bf16(attention_3 / denominator);
  }

  let cache_word_base = (
    current_slot * KV_SIZE + kv_head * HEAD_DIM
  ) >> 1u;
  if (lane >= COMPUTE_LANES) {
    let loader_lane = lane - COMPUTE_LANES;
    let first_word = loader_lane * 2u;
    let second_word = first_word + 1u;
    cached_value_words[first_word] = value_cache[
      params.value_cache_offset + cache_word_base + first_word
    ];
    cached_value_words[second_word] = value_cache[
      params.value_cache_offset + cache_word_base + second_word
    ];
  }
  workgroupBarrier();

  var current_value = vec4<f32>(0.0f);
  if (is_active) {
    let first_word = head_lane * 2u;
    current_value = unpack_bf16_quad(
      cached_value_words[first_word],
      cached_value_words[first_word + 1u],
    );
  }
  if (relative_head == 0u) {
    reduction[component_base] = current_value.x * current_value.x;
    reduction[component_base + 1u] = current_value.y * current_value.y;
    reduction[component_base + 2u] = current_value.z * current_value.z;
    reduction[component_base + 3u] = current_value.w * current_value.w;
  }
  reduce_first_plane(lane);

  var direction_0 = 0.0f;
  var direction_1 = 0.0f;
  var direction_2 = 0.0f;
  var direction_3 = 0.0f;
  if (is_active) {
    let magnitude = max(sqrt(reduction[0]), XSA_EPSILON);
    direction_0 = current_value.x / magnitude;
    direction_1 = current_value.y / magnitude;
    direction_2 = current_value.z / magnitude;
    direction_3 = current_value.w / magnitude;
  }
  workgroupBarrier();

  if (is_active) {
    let reduction_base = relative_head * HEAD_DIM + component_base;
    reduction[reduction_base] = bf16_attention_0 * direction_0;
    reduction[reduction_base + 1u] = bf16_attention_1 * direction_1;
    reduction[reduction_base + 2u] = bf16_attention_2 * direction_2;
    reduction[reduction_base + 3u] = bf16_attention_3 * direction_3;
  }
  reduce_head_planes(relative_head, head_lane);

  if (is_active) {
    let dot = reduction[relative_head * HEAD_DIM];
    let alpha = xsa_alpha[params.xsa_offset + query_head];
    let correction_0 = to_bf16(alpha * dot * direction_0);
    let correction_1 = to_bf16(alpha * dot * direction_1);
    let correction_2 = to_bf16(alpha * dot * direction_2);
    let correction_3 = to_bf16(alpha * dot * direction_3);
    let xsa_value_0 = to_bf16(bf16_attention_0 - correction_0);
    let xsa_value_1 = to_bf16(bf16_attention_1 - correction_1);
    let xsa_value_2 = to_bf16(bf16_attention_2 - correction_2);
    let xsa_value_3 = to_bf16(bf16_attention_3 - correction_3);
    let gate_logit = projected[
      params.projected_offset + ATTENTION_GATE_OFFSET + query_head
    ];
    // Preserve the official BF16 sigmoid and scalar-multiply boundaries.
    let sigmoid_gate = exact_bf16_sigmoid(gate_logit);
    let gate = to_bf16(2.0f * sigmoid_gate);
    let output_base = params.output_offset + query_head * HEAD_DIM + component_base;
    output[output_base] = to_bf16(xsa_value_0 * gate);
    output[output_base + 1u] = to_bf16(xsa_value_1 * gate);
    output[output_base + 2u] = to_bf16(xsa_value_2 * gate);
    output[output_base + 3u] = to_bf16(xsa_value_3 * gate);
  }
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
  lookup_offset: u32,
  _padding_1: u32,
  _padding_2: u32,
  _padding_3: u32,
  _padding_4: u32,
}

@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SwiGluParams;
@group(0) @binding(3) var<storage, read> silu_by_bf16: array<f32>;

${BF16_ROUND_WGSL}

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
  // The projection boundary stores an f32 representation of an exact BF16,
  // so its upper word is the complete, lossless lookup key.
  let silu = silu_by_bf16[
    params.lookup_offset + (bitcast<u32>(gate) >> 16u)
  ];
  output[params.output_offset + index] = to_bf16(silu * up);
}
`;

export const SAMPLER_WORKGROUP_SIZE = 256;
export const SAMPLER_ENTRIES_PER_PARTITION = SAMPLER_WORKGROUP_SIZE;

/**
 * Fully GPU-resident full-vocabulary nucleus sampler.
 *
 * `prepare_nucleus_partitions` applies Violetto's sigmoid softcap and
 * temperature, bitonic-sorts every 256-token partition, and writes all sorted
 * weights, token ids, and inclusive prefix masses. `sample_nucleus` finds the
 * exact positive-f32 probability threshold whose globally sorted cumulative
 * mass first reaches top-p. It then advances a persistent xorshift32 state and
 * samples directly from that nucleus. No top-k truncation is applied, and no
 * logits or random values cross to the CPU.
 *
 * Scores are ordered by descending f32 weight, with ascending token id as the
 * deterministic tie-break. If top-p cuts through equal scores, only the first
 * ids in that ordering are retained. Invalid, NaN, infinite, and padded-vocab
 * logits are excluded before sorting.
 *
 * Bindings (seven production storage buffers, below WebGPU's guaranteed
 * minimum of eight):
 *   0: raw f32 logits
 *   1: partition-sorted probability weights, read_write f32
 *   2: partition-sorted token ids, read_write u32
 *   3: inclusive partition prefix masses, read_write f32
 *   4: next-token ids, read_write u32
 *   5: persistent RNG state, read_write u32
 *   6: SamplerParams uniform (48 bytes)
 *   7: GPU-initialized BF16 sampling-weight table, read f32
 *
 * The one-time lookup initializer writes binding 8. Production dispatches use
 * only the immutable binding-7 view of the same 256 KiB buffer.
 *
 * Params layout:
 *   u32 vocab_size, logits_offset, scratch_offset, scratch_count;
 *   u32 token_output_offset, rng_offset, partition_count, _padding_0;
 *   f32 temperature, top_p; u32 _padding_1, _padding_2.
 *
 * Dispatch `prepare_nucleus_partitions` with
 * `(ceil(vocab_size / 256), 1, 1)`, then `sample_nucleus` with `(1, 1, 1)`.
 * For Limite these are 593 and 1. Each scratch buffer needs
 * `partition_count * 256` entries (151808 for the tokenizer vocabulary).
 */
export const GPU_TOP_P_SAMPLER_WGSL = /* wgsl */ `
const WORKGROUP_SIZE = ${SAMPLER_WORKGROUP_SIZE}u;
const PARTITION_SIZE = ${SAMPLER_ENTRIES_PER_PARTITION}u;
const NEGATIVE_INFINITY = -3.402823e38f;
const EOS_TOKEN = 151643u;

struct SamplerParams {
  vocab_size: u32,
  logits_offset: u32,
  scratch_offset: u32,
  scratch_count: u32,
  token_output_offset: u32,
  rng_offset: u32,
  partition_count: u32,
  sampling_weight_offset: u32,
  temperature: f32,
  top_p: f32,
  _padding_1: u32,
  _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> sorted_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> sorted_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> prefix_masses: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_token: array<u32>;
@group(0) @binding(5) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(6) var<uniform> params: SamplerParams;
@group(0) @binding(7) var<storage, read> sampling_weight_by_bf16: array<f32>;

var<workgroup> sort_values: array<f32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> sort_ids: array<u32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> scan_values: array<f32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> reduce_values: array<f32, ${SAMPLER_WORKGROUP_SIZE}>;
var<workgroup> partition_counts: array<u32, ${Math.ceil(LIMITE_TOKENIZER_VOCAB_SIZE / SAMPLER_WORKGROUP_SIZE)}>;
var<workgroup> partition_masses: array<f32, ${Math.ceil(LIMITE_TOKENIZER_VOCAB_SIZE / SAMPLER_WORKGROUP_SIZE)}>;
var<workgroup> search_bits: u32;
var<workgroup> search_low: u32;
var<workgroup> search_high: u32;
var<workgroup> total_mass: f32;
var<workgroup> target_mass: f32;
var<workgroup> cutoff_partition: u32;
var<workgroup> cutoff_count: u32;

fn better(value_a: f32, id_a: u32, value_b: f32, id_b: u32) -> bool {
  return value_a > value_b || (value_a == value_b && id_a < id_b);
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
fn prepare_nucleus_partitions(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let partition_index = group.x;
  if (partition_index >= params.partition_count) {
    return;
  }
  let token = partition_index * WORKGROUP_SIZE + lane;
  var value = 0.0f;
  var id = 0xffffffffu;
  if (token < params.vocab_size) {
    let candidate = logits[params.logits_offset + token];
    // Both comparisons are false for NaN; either rejects the corresponding
    // infinity while accepting every finite f32 value.
    if (candidate >= NEGATIVE_INFINITY && candidate <= -NEGATIVE_INFINITY) {
      // The exact tied head rounds each logit to BF16 before sampling.
      value = sampling_weight_by_bf16[
        params.sampling_weight_offset + (bitcast<u32>(candidate) >> 16u)
      ];
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

  // Reverse the ascending bitonic order into best-first order. An inclusive
  // Hillis-Steele scan produces one local mass prefix per sorted token.
  let source = WORKGROUP_SIZE - 1u - lane;
  let weight = sort_values[source];
  let sorted_id = sort_ids[source];
  scan_values[lane] = weight;
  workgroupBarrier();
  var offset = 1u;
  while (offset < WORKGROUP_SIZE) {
    var addend = 0.0f;
    if (lane >= offset) {
      addend = scan_values[lane - offset];
    }
    workgroupBarrier();
    scan_values[lane] += addend;
    workgroupBarrier();
    offset <<= 1u;
  }

  let destination = params.scratch_offset + partition_index * PARTITION_SIZE + lane;
  sorted_weights[destination] = weight;
  sorted_ids[destination] = sorted_id;
  prefix_masses[destination] = scan_values[lane];
}

// Number of sorted values whose positive-f32 bit representation is at least
// threshold_bits. For positive finite f32 values, integer and numeric order
// are identical. A binary search therefore gives an exact score boundary.
fn count_at_least(partition_index: u32, threshold_bits: u32) -> u32 {
  let base = params.scratch_offset + partition_index * PARTITION_SIZE;
  var low = 0u;
  var high = PARTITION_SIZE;
  while (low < high) {
    let middle = low + ((high - low) >> 1u);
    let bits = bitcast<u32>(sorted_weights[base + middle]);
    if (bits >= threshold_bits) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}

fn count_greater(partition_index: u32, threshold_bits: u32) -> u32 {
  if (threshold_bits == 0xffffffffu) {
    return 0u;
  }
  return count_at_least(partition_index, threshold_bits + 1u);
}

fn mass_for_count(partition_index: u32, count: u32) -> f32 {
  if (count == 0u) {
    return 0.0f;
  }
  let base = params.scratch_offset + partition_index * PARTITION_SIZE;
  return prefix_masses[base + count - 1u];
}

fn reduce_sum(lane: u32, value: f32) -> f32 {
  reduce_values[lane] = value;
  workgroupBarrier();
  var stride = WORKGROUP_SIZE >> 1u;
  while (stride > 0u) {
    if (lane < stride) {
      reduce_values[lane] += reduce_values[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  return reduce_values[0];
}

fn lane_mass_at_least(lane: u32, threshold_bits: u32) -> f32 {
  var value = 0.0f;
  var partition_index = lane;
  while (partition_index < params.partition_count) {
    value += mass_for_count(
      partition_index,
      count_at_least(partition_index, threshold_bits),
    );
    partition_index += WORKGROUP_SIZE;
  }
  return value;
}

@compute @workgroup_size(${SAMPLER_WORKGROUP_SIZE})
fn sample_nucleus(@builtin(local_invocation_index) lane: u32) {
  // The last prefix is each partition's complete mass. Invalid tail entries
  // have weight zero, so using index 255 is valid for the final partition.
  var lane_total = 0.0f;
  var partition_index = lane;
  while (partition_index < params.partition_count) {
    lane_total += mass_for_count(partition_index, PARTITION_SIZE);
    partition_index += WORKGROUP_SIZE;
  }
  let all_mass = reduce_sum(lane, lane_total);
  if (lane == 0u) {
    total_mass = all_mass;
    target_mass = clamp(params.top_p, 0.0f, 1.0f) * all_mass;
    search_low = 0u;
    // +infinity is an exclusive upper bound for all finite positive weights.
    search_high = 0x7f800000u;
  }
  workgroupBarrier();

  // Find the greatest representable positive f32 threshold whose retained
  // mass is at least target_mass. Since weights are stored f32 values, this is
  // exactly the boundary score in the global descending order, not a binning
  // or approximate quantile.
  // 0x7f800000 fits within 31 binary decisions. A fixed iteration count keeps
  // every barrier in statically uniform control flow on all WGSL validators.
  for (var iteration = 0u; iteration < 31u; iteration += 1u) {
    if (lane == 0u) {
      search_bits = search_low + ((search_high - search_low) >> 1u);
    }
    workgroupBarrier();
    let mass = reduce_sum(lane, lane_mass_at_least(lane, search_bits));
    if (lane == 0u) {
      if (mass >= target_mass) {
        search_low = search_bits;
      } else {
        search_high = search_bits;
      }
    }
    workgroupBarrier();
  }

  let threshold_bits = search_low;
  let threshold = bitcast<f32>(threshold_bits);
  var lane_higher_mass = 0.0f;
  var lane_equal_count = 0u;
  partition_index = lane;
  while (partition_index < params.partition_count) {
    let greater_count = count_greater(partition_index, threshold_bits);
    let at_least_count = count_at_least(partition_index, threshold_bits);
    lane_higher_mass += mass_for_count(partition_index, greater_count);
    let equal_count = at_least_count - greater_count;
    partition_counts[partition_index] = equal_count;
    lane_equal_count += equal_count;
    partition_index += WORKGROUP_SIZE;
  }
  let higher_mass = reduce_sum(lane, lane_higher_mass);
  let equal_count = u32(reduce_sum(lane, f32(lane_equal_count)));

  if (lane == 0u) {
    // Find the smallest number of boundary ties that reaches the target using
    // the same f32 multiplication used to define their retained mass. A small
    // integer binary search avoids a divide/ceil rounding error at exact
    // floating-point boundaries.
    let residual = max(target_mass - higher_mass, 0.0f);
    var required_low = 1u;
    var required_high = max(equal_count, 1u);
    while (required_low < required_high) {
      let middle = required_low + ((required_high - required_low) >> 1u);
      if (f32(middle) * threshold >= residual) {
        required_high = middle;
      } else {
        required_low = middle + 1u;
      }
    }
    let required = required_low;

    var remaining = required;
    cutoff_partition = 0xffffffffu;
    cutoff_count = 0u;
    for (var index = 0u; index < params.partition_count; index += 1u) {
      let count = partition_counts[index];
      if (remaining > count) {
        remaining -= count;
      } else if (cutoff_partition == 0xffffffffu) {
        cutoff_partition = index;
        cutoff_count = remaining;
      }
    }
  }
  workgroupBarrier();

  // Build each partition's exact retained mass. Scores above the threshold
  // are always kept. Equal scores are kept through the global ascending-id
  // tie boundary, matching Paradigma's vLLM duplicate-logit pass.
  partition_index = lane;
  while (partition_index < params.partition_count) {
    let greater_count = count_greater(partition_index, threshold_bits);
    var kept_count = greater_count;
    if (partition_index < cutoff_partition) {
      kept_count += partition_counts[partition_index];
    } else if (partition_index == cutoff_partition) {
      kept_count += cutoff_count;
    }
    partition_counts[partition_index] = kept_count;
    partition_masses[partition_index] = mass_for_count(partition_index, kept_count);
    partition_index += WORKGROUP_SIZE;
  }
  workgroupBarrier();

  if (lane != 0u) {
    return;
  }
  if (!(total_mass > 0.0f) || cutoff_partition == 0xffffffffu) {
    next_token[params.token_output_offset] = EOS_TOKEN;
    return;
  }

  var kept_total = 0.0f;
  for (var index = 0u; index < params.partition_count; index += 1u) {
    kept_total += partition_masses[index];
  }
  var pick = random_unit() * kept_total;
  // cutoff_partition always has at least one retained boundary token and is a
  // safe positive-mass fallback if f32 subtraction rounds past every interval.
  var selected_partition = cutoff_partition;
  for (var index = 0u; index < params.partition_count; index += 1u) {
    let mass = partition_masses[index];
    if (pick < mass) {
      selected_partition = index;
      break;
    }
    pick -= mass;
  }

  let selected_count = partition_counts[selected_partition];
  if (selected_count == 0u) {
    next_token[params.token_output_offset] = EOS_TOKEN;
    return;
  }
  let base = params.scratch_offset + selected_partition * PARTITION_SIZE;
  var low = 0u;
  var high = selected_count;
  while (low < high) {
    let middle = low + ((high - low) >> 1u);
    if (prefix_masses[base + middle] > pick) {
      high = middle;
    } else {
      low = middle + 1u;
    }
  }
  let selected = min(low, selected_count - 1u);
  next_token[params.token_output_offset] = sorted_ids[base + selected];
}
`;
