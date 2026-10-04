import { LIMITE_HIDDEN_SIZE } from "./ops";

export const VECTOR_WORKGROUP_SIZE = 256;

export const COPY_VECTOR_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;

@compute @workgroup_size(${VECTOR_WORKGROUP_SIZE})
fn copy_vector(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index < ${LIMITE_HIDDEN_SIZE}u) {
    destination[index] = source[index];
  }
}
`;

/**
 * Mixes two hidden vectors and emits both the raw result and its gain-free
 * RMS-normalized form. Keeping the two writes together removes one full
 * hidden-vector pass at every residual boundary.
 */
export const MIX_AND_NORM_WGSL = /* wgsl */ `
const WIDTH = ${LIMITE_HIDDEN_SIZE}u;
const EPSILON = ${1 / 128}f;

struct MixParams {
  left_scale: f32,
  right_scale: f32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> mixed: array<f32>;
@group(0) @binding(3) var<storage, read_write> normalized: array<f32>;
@group(0) @binding(4) var<uniform> params: MixParams;

var<workgroup> squared_sum: array<f32, ${VECTOR_WORKGROUP_SIZE}>;

@compute @workgroup_size(${VECTOR_WORKGROUP_SIZE})
fn mix_and_norm(@builtin(local_invocation_index) lane: u32) {
  var sum = 0.0f;
  for (var index = lane; index < WIDTH; index += ${VECTOR_WORKGROUP_SIZE}u) {
    let value = params.left_scale * left[index]
      + params.right_scale * right[index];
    mixed[index] = value;
    sum += value * value;
  }
  squared_sum[lane] = sum;
  workgroupBarrier();

  var stride = ${VECTOR_WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      squared_sum[lane] += squared_sum[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }

  let inverse_rms = inverseSqrt(squared_sum[0] / f32(WIDTH) + EPSILON);
  for (var index = lane; index < WIDTH; index += ${VECTOR_WORKGROUP_SIZE}u) {
    normalized[index] = mixed[index] * inverse_rms;
  }
}
`;

/**
 * Violetto's two MUDD junctions. The five small MUDD tensors are expanded to
 * f32 once while loading and kept in one GPU buffer. One workgroup computes
 * the exact graph: RMS -> dense1 -> GELU -> two 3-way mixtures -> RMS.
 */
export const MUDD_WGSL = /* wgsl */ `
const WIDTH = ${LIMITE_HIDDEN_SIZE}u;
const INNER = 32u;
const TAPS = 3u;
const EPSILON = ${1 / 128}f;
const INV_SQRT_2 = 0.7071067811865475244f;

struct MuddParams {
  layer: u32,
  dense1_offset: u32,
  dense2_offset: u32,
  dense2_mlp_offset: u32,
  bias_offset: u32,
  bias_mlp_offset: u32,
  _padding_0: u32,
  _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> current: array<f32>;
@group(0) @binding(2) var<storage, read> tap_zero: array<f32>;
@group(0) @binding(3) var<storage, read> tap_middle: array<f32>;
@group(0) @binding(4) var<storage, read_write> attention_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> residual_base: array<f32>;
@group(0) @binding(6) var<uniform> params: MuddParams;

var<workgroup> reduction: array<f32, ${VECTOR_WORKGROUP_SIZE}>;
var<workgroup> inner_values: array<f32, 32>;
var<workgroup> mixture: array<f32, 6>;
var<workgroup> inverse_current_rms: f32;
var<workgroup> inverse_attention_rms: f32;

// Maximum absolute error is below 1.5e-7 over the range that matters here.
fn erf_approx(value: f32) -> f32 {
  let sign_value = select(-1.0f, 1.0f, value >= 0.0f);
  let x = abs(value);
  let t = 1.0f / (1.0f + 0.3275911f * x);
  let polynomial = (((((1.061405429f * t - 1.453152027f) * t)
    + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  return sign_value * (1.0f - polynomial * exp(-x * x));
}

fn reduce_sum(lane: u32, value: f32) -> f32 {
  reduction[lane] = value;
  workgroupBarrier();
  var stride = ${VECTOR_WORKGROUP_SIZE / 2}u;
  while (stride > 0u) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    workgroupBarrier();
    stride >>= 1u;
  }
  return reduction[0];
}

@compute @workgroup_size(${VECTOR_WORKGROUP_SIZE})
fn mudd(@builtin(local_invocation_index) lane: u32) {
  var local_square_sum = 0.0f;
  for (var index = lane; index < WIDTH; index += ${VECTOR_WORKGROUP_SIZE}u) {
    let value = current[index];
    local_square_sum += value * value;
  }
  let current_square_sum = reduce_sum(lane, local_square_sum);
  if (lane == 0u) {
    inverse_current_rms = inverseSqrt(current_square_sum / f32(WIDTH) + EPSILON);
  }
  workgroupBarrier();

  if (lane < INNER) {
    var projected = 0.0f;
    let row = params.dense1_offset + lane * WIDTH;
    for (var column = 0u; column < WIDTH; column += 1u) {
      projected += weights[row + column] * current[column] * inverse_current_rms;
    }
    inner_values[lane] = 0.5f * projected
      * (1.0f + erf_approx(projected * INV_SQRT_2));
  }
  workgroupBarrier();

  if (lane < 6u) {
    let is_mlp = lane >= 3u;
    let tap = lane % 3u;
    let dense_offset = select(
      params.dense2_offset,
      params.dense2_mlp_offset,
      is_mlp,
    ) + (params.layer * TAPS + tap) * INNER;
    let bias_offset = select(params.bias_offset, params.bias_mlp_offset, is_mlp)
      + params.layer * TAPS + tap;
    var value = weights[bias_offset];
    for (var index = 0u; index < INNER; index += 1u) {
      value += inner_values[index] * weights[dense_offset + index];
    }
    mixture[lane] = value;
  }
  workgroupBarrier();

  var attention_square_sum = 0.0f;
  for (var index = lane; index < WIDTH; index += ${VECTOR_WORKGROUP_SIZE}u) {
    // Preserve the checkpoint's left-to-right accumulation order.
    let attention = tap_zero[index] * mixture[0]
      + tap_middle[index] * mixture[1]
      + current[index] * mixture[2];
    let residual = tap_zero[index] * mixture[3]
      + tap_middle[index] * mixture[4]
      + current[index] * mixture[5];
    attention_input[index] = attention;
    residual_base[index] = residual;
    attention_square_sum += attention * attention;
  }
  let total_attention_square = reduce_sum(lane, attention_square_sum);
  if (lane == 0u) {
    inverse_attention_rms = inverseSqrt(
      total_attention_square / f32(WIDTH) + EPSILON,
    );
  }
  workgroupBarrier();
  for (var index = lane; index < WIDTH; index += ${VECTOR_WORKGROUP_SIZE}u) {
    attention_input[index] *= inverse_attention_rms;
  }
}
`;
