export const PACKED_BF16_MATVEC_WORKGROUP_SIZE = 64;
export const PACKED_BF16_MATVEC_ROWS_PER_GROUP = 8;

/**
 * WGSL helper for representing BF16 tensors in f32 storage.
 *
 * The integer bias implements round-to-nearest-even. NaNs and infinities are
 * returned unchanged so rounding cannot turn a NaN payload into an infinity.
 */
export const BF16_ROUND_WGSL = /* wgsl */ `
fn to_bf16(value: f32) -> f32 {
  let bits = bitcast<u32>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return value;
  }
  let rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
  return bitcast<f32>(rounded & 0xffff0000u);
}
`;

/**
 * Matvec for the exact packed-BF16 tied token embedding / vocabulary head.
 * Every workgroup produces eight vocabulary rows while preserving each
 * checkpoint BF16 value exactly on load. Dot products accumulate in f32 and
 * are rounded once at the model's BF16 projection boundary.
 */
export const PACKED_BF16_MATVEC_WGSL = /* wgsl */ `
struct PackedBf16MatVecParams {
  row_count: u32,
  column_count: u32,
  output_element_offset: u32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
  _padding_3: u32,
  _padding_4: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: PackedBf16MatVecParams;

${BF16_ROUND_WGSL}

var<workgroup> partial_0: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_1: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_2: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_3: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_4: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_5: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_6: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;
var<workgroup> partial_7: array<f32, ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}>;

fn unpack_bf16_pair(word: u32) -> vec2<f32> {
  return vec2<f32>(
    bitcast<f32>(word << 16u),
    bitcast<f32>(word & 0xffff0000u),
  );
}

fn unpack_bf16x4(first: u32, second: u32) -> vec4<f32> {
  let low = unpack_bf16_pair(first);
  let high = unpack_bf16_pair(second);
  return vec4<f32>(low.x, low.y, high.x, high.y);
}

@compute @workgroup_size(${PACKED_BF16_MATVEC_WORKGROUP_SIZE})
fn packed_bf16_matvec(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
  let row = workgroup_id.x * ${PACKED_BF16_MATVEC_ROWS_PER_GROUP}u;
  if (row >= params.row_count) {
    return;
  }

  let vectors_per_row = params.column_count / 4u;
  let words_per_row = params.column_count / 2u;
  var sum_0 = 0.0f;
  var sum_1 = 0.0f;
  var sum_2 = 0.0f;
  var sum_3 = 0.0f;
  var sum_4 = 0.0f;
  var sum_5 = 0.0f;
  var sum_6 = 0.0f;
  var sum_7 = 0.0f;

  for (var vector = lane; vector < vectors_per_row; vector += ${PACKED_BF16_MATVEC_WORKGROUP_SIZE}u) {
    let activation = input[vector];
    let word_in_row = vector * 2u;
    let word_0 = row * words_per_row + word_in_row;
    sum_0 += dot(activation, unpack_bf16x4(matrix[word_0], matrix[word_0 + 1u]));

    if (row + 1u < params.row_count) {
      let word_1 = word_0 + words_per_row;
      sum_1 += dot(activation, unpack_bf16x4(matrix[word_1], matrix[word_1 + 1u]));
    }

    if (row + 2u < params.row_count) {
      let word_2 = word_0 + 2u * words_per_row;
      sum_2 += dot(activation, unpack_bf16x4(matrix[word_2], matrix[word_2 + 1u]));
    }

    if (row + 3u < params.row_count) {
      let word_3 = word_0 + 3u * words_per_row;
      sum_3 += dot(activation, unpack_bf16x4(matrix[word_3], matrix[word_3 + 1u]));
    }

    if (row + 4u < params.row_count) {
      let word_4 = word_0 + 4u * words_per_row;
      sum_4 += dot(activation, unpack_bf16x4(matrix[word_4], matrix[word_4 + 1u]));
    }

    if (row + 5u < params.row_count) {
      let word_5 = word_0 + 5u * words_per_row;
      sum_5 += dot(activation, unpack_bf16x4(matrix[word_5], matrix[word_5 + 1u]));
    }

    if (row + 6u < params.row_count) {
      let word_6 = word_0 + 6u * words_per_row;
      sum_6 += dot(activation, unpack_bf16x4(matrix[word_6], matrix[word_6 + 1u]));
    }

    if (row + 7u < params.row_count) {
      let word_7 = word_0 + 7u * words_per_row;
      sum_7 += dot(activation, unpack_bf16x4(matrix[word_7], matrix[word_7 + 1u]));
    }
  }

  partial_0[lane] = sum_0;
  partial_1[lane] = sum_1;
  partial_2[lane] = sum_2;
  partial_3[lane] = sum_3;
  partial_4[lane] = sum_4;
  partial_5[lane] = sum_5;
  partial_6[lane] = sum_6;
  partial_7[lane] = sum_7;
  workgroupBarrier();

  var stride = ${PACKED_BF16_MATVEC_WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      partial_0[lane] += partial_0[lane + stride];
      partial_1[lane] += partial_1[lane + stride];
      partial_2[lane] += partial_2[lane + stride];
      partial_3[lane] += partial_3[lane + stride];
      partial_4[lane] += partial_4[lane + stride];
      partial_5[lane] += partial_5[lane + stride];
      partial_6[lane] += partial_6[lane + stride];
      partial_7[lane] += partial_7[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  if (lane == 0u) {
    output[params.output_element_offset + row] = to_bf16(partial_0[0]);
    if (row + 1u < params.row_count) {
      output[params.output_element_offset + row + 1u] = to_bf16(partial_1[0]);
    }
    if (row + 2u < params.row_count) {
      output[params.output_element_offset + row + 2u] = to_bf16(partial_2[0]);
    }
    if (row + 3u < params.row_count) {
      output[params.output_element_offset + row + 3u] = to_bf16(partial_3[0]);
    }
    if (row + 4u < params.row_count) {
      output[params.output_element_offset + row + 4u] = to_bf16(partial_4[0]);
    }
    if (row + 5u < params.row_count) {
      output[params.output_element_offset + row + 5u] = to_bf16(partial_5[0]);
    }
    if (row + 6u < params.row_count) {
      output[params.output_element_offset + row + 6u] = to_bf16(partial_6[0]);
    }
    if (row + 7u < params.row_count) {
      output[params.output_element_offset + row + 7u] = to_bf16(partial_7[0]);
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

${BF16_ROUND_WGSL}

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
    output[output_base + column] = to_bf16(
      input[input_base + column] * inverse_rms,
    );
  }
}
`;
