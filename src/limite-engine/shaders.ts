/** Number of signed int4 weights in one block of the production artifact. */
export const Q4_32_BLOCK_ELEMENTS = 32;

/** Four u32 words hold the 32 low-nibble-first int4 codes in each block. */
export const Q4_32_QUANT_WORDS_PER_BLOCK = 4;

export const Q4_32_MATVEC_WORKGROUP_SIZE = 64;
export const Q4_32_MATVEC_ROWS_PER_GROUP = 4;

/**
 * Decode matvec for Kalkulator's compact symmetric Q4 artifact.
 *
 * The storage buffer contains two contiguous regions: first all matrix codes,
 * as four u32 words per 32-weight block, then one f16 scale per block packed
 * two-per-u32 at `scale_offset_words`. Codes are stored low nibble first as
 * `(signed_value + 8)`, giving the dequantization `(code - 8) * scale`.
 *
 * Bindings:
 *   0: packed codes and scales in one array<u32>
 *   1: f32 input vector
 *   2: f32 output vector
 *   3: Q4MatVecParams uniform (16 bytes)
 *
 * Q4MatVecParams is four little-endian u32 values:
 *   [row_count, column_count, scale_offset_words, output_element_offset]
 *
 * Four adjacent output rows share every activation load. Each lane owns one
 * packed u32 (eight adjacent weights), so all matrix and scale reads are
 * coalesced instead of redundantly loading the same word from eight lanes.
 * Two vec4 dot products accumulate each word before applying its block scale.
 * Weights, accumulation, and output retain the same Q4/FP32 contract.
 * Dispatch `ceil(row_count / 4)` workgroups. `column_count` must be a non-zero
 * multiple of 32, and `scale_offset_words` must be the aligned word offset of
 * the first packed scale.
 */
export const Q4_32_MATVEC_WGSL = /* wgsl */ `
struct Q4MatVecParams {
  row_count: u32,
  column_count: u32,
  scale_offset_words: u32,
  output_element_offset: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Q4MatVecParams;

var<workgroup> partial_0: array<f32, ${Q4_32_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_1: array<f32, ${Q4_32_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_2: array<f32, ${Q4_32_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_3: array<f32, ${Q4_32_MATVEC_WORKGROUP_SIZE}>;

fn block_scale(block_index: u32) -> f32 {
  let packed = unpack2x16float(matrix[params.scale_offset_words + (block_index >> 1u)]);
  return select(packed.x, packed.y, (block_index & 1u) != 0u);
}

fn dot_word(packed: u32, low: vec4<f32>, high: vec4<f32>) -> f32 {
  let shifts = vec4<u32>(0u, 4u, 8u, 12u);
  let codes_low = vec4<f32>((vec4<u32>(packed) >> shifts) & vec4<u32>(15u)) - vec4<f32>(8.0f);
  let codes_high = vec4<f32>((vec4<u32>(packed >> 16u) >> shifts) & vec4<u32>(15u)) - vec4<f32>(8.0f);
  return dot(low, codes_low) + dot(high, codes_high);
}

@compute @workgroup_size(${Q4_32_MATVEC_WORKGROUP_SIZE})
fn q4_32_matvec(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let row = workgroup_id.x * ${Q4_32_MATVEC_ROWS_PER_GROUP}u;
  if (row >= params.row_count) {
    return;
  }

  let blocks_per_row = params.column_count / ${Q4_32_BLOCK_ELEMENTS}u;
  var sum_0 = 0.0f;
  var sum_1 = 0.0f;
  var sum_2 = 0.0f;
  var sum_3 = 0.0f;

  const blocks_per_wave = ${Q4_32_MATVEC_WORKGROUP_SIZE / Q4_32_QUANT_WORDS_PER_BLOCK}u;
  let lane_block = lane / ${Q4_32_QUANT_WORDS_PER_BLOCK}u;
  let word_in_block = lane % ${Q4_32_QUANT_WORDS_PER_BLOCK}u;
  for (var wave = 0u; wave < blocks_per_row; wave += blocks_per_wave) {
    let block = wave + lane_block;
    if (block >= blocks_per_row) {
      continue;
    }
    let block_0 = row * blocks_per_row + block;
    let packed_0 = matrix[
      block_0 * ${Q4_32_QUANT_WORDS_PER_BLOCK}u + word_in_block
    ];
    let scale_0 = block_scale(block_0);

    var packed_1 = 0u;
    var scale_1 = 0.0f;
    if (row + 1u < params.row_count) {
      let block_1 = block_0 + blocks_per_row;
      packed_1 = matrix[
        block_1 * ${Q4_32_QUANT_WORDS_PER_BLOCK}u + word_in_block
      ];
      scale_1 = block_scale(block_1);
    }

    var packed_2 = 0u;
    var scale_2 = 0.0f;
    if (row + 2u < params.row_count) {
      let block_2 = block_0 + 2u * blocks_per_row;
      packed_2 = matrix[
        block_2 * ${Q4_32_QUANT_WORDS_PER_BLOCK}u + word_in_block
      ];
      scale_2 = block_scale(block_2);
    }

    var packed_3 = 0u;
    var scale_3 = 0.0f;
    if (row + 3u < params.row_count) {
      let block_3 = block_0 + 3u * blocks_per_row;
      packed_3 = matrix[
        block_3 * ${Q4_32_QUANT_WORDS_PER_BLOCK}u + word_in_block
      ];
      scale_3 = block_scale(block_3);
    }

    let input_base = block * ${Q4_32_BLOCK_ELEMENTS}u + word_in_block * 8u;
    let low = vec4<f32>(input[input_base], input[input_base + 1u], input[input_base + 2u], input[input_base + 3u]);
    let high = vec4<f32>(input[input_base + 4u], input[input_base + 5u], input[input_base + 6u], input[input_base + 7u]);
    sum_0 += dot_word(packed_0, low, high) * scale_0;
    sum_1 += dot_word(packed_1, low, high) * scale_1;
    sum_2 += dot_word(packed_2, low, high) * scale_2;
    sum_3 += dot_word(packed_3, low, high) * scale_3;
  }

  partial_0[lane] = sum_0;
  partial_1[lane] = sum_1;
  partial_2[lane] = sum_2;
  partial_3[lane] = sum_3;
  workgroupBarrier();

  var stride = ${Q4_32_MATVEC_WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      partial_0[lane] += partial_0[lane + stride];
      partial_1[lane] += partial_1[lane + stride];
      partial_2[lane] += partial_2[lane + stride];
      partial_3[lane] += partial_3[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  if (lane == 0u) {
    output[params.output_element_offset + row] = partial_0[0];
    if (row + 1u < params.row_count) {
      output[params.output_element_offset + row + 1u] = partial_1[0];
    }
    if (row + 2u < params.row_count) {
      output[params.output_element_offset + row + 2u] = partial_2[0];
    }
    if (row + 3u < params.row_count) {
      output[params.output_element_offset + row + 3u] = partial_3[0];
    }
  }
}
`;

export const RMS_NORM_F32_WORKGROUP_SIZE = 256;

/**
 * Gain-free RMS normalization for one or more contiguous f32 rows.
 *
 * Bindings:
 *   0: f32 input rows
 *   1: f32 output rows
 *   2: RmsNormParams uniform (32 bytes)
 *
 * RmsNormParams layout:
 *   u32 row_count, row_width, input_element_offset, output_element_offset;
 *   f32 epsilon;
 *   u32 padding[3];
 *
 * Dispatch exactly (row_count, 1, 1). This shader deliberately has no gain
 * buffer: Limite's RMS normalizations are gain-free. Input and output bindings
 * should not overlap; use ping-pong activation buffers.
 */
export const RMS_NORM_F32_WGSL = /* wgsl */ `
struct RmsNormParams {
  row_count: u32,
  row_width: u32,
  input_element_offset: u32,
  output_element_offset: u32,
  epsilon: f32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: RmsNormParams;

var<workgroup> squared_sum: array<f32, ${RMS_NORM_F32_WORKGROUP_SIZE}>;

@compute @workgroup_size(${RMS_NORM_F32_WORKGROUP_SIZE})
fn rms_norm_f32(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let row = workgroup_id.x;
  if (row >= params.row_count) {
    return;
  }

  let input_base = params.input_element_offset + row * params.row_width;
  let output_base = params.output_element_offset + row * params.row_width;
  var sum = 0.0f;
  for (var column = lane; column < params.row_width; column += ${RMS_NORM_F32_WORKGROUP_SIZE}u) {
    let value = input[input_base + column];
    sum += value * value;
  }

  squared_sum[lane] = sum;
  workgroupBarrier();

  var stride = ${RMS_NORM_F32_WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      squared_sum[lane] += squared_sum[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  let inverse_rms = inverseSqrt(squared_sum[0] / f32(params.row_width) + params.epsilon);
  for (var column = lane; column < params.row_width; column += ${RMS_NORM_F32_WORKGROUP_SIZE}u) {
    output[output_base + column] = input[input_base + column] * inverse_rms;
  }
}
`;
