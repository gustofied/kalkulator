import { LIMITE_EOS_TOKEN, LIMITE_TOKENIZER_VOCAB_SIZE } from "../limite-config";

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
 * Bindings (six storage buffers):
 *   0: raw f32 logits
 *   1: partition-sorted probability weights, read_write f32
 *   2: partition-sorted token ids, read_write u32
 *   3: inclusive partition prefix masses, read_write f32
 *   4: next-token ids, read_write u32
 *   5: persistent RNG state, read_write u32
 *   6: SamplerParams uniform (48 bytes)
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
const EOS_TOKEN = ${LIMITE_EOS_TOKEN}u;

struct SamplerParams {
  vocab_size: u32,
  logits_offset: u32,
  scratch_offset: u32,
  scratch_count: u32,
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
@group(0) @binding(1) var<storage, read_write> sorted_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> sorted_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> prefix_masses: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_token: array<u32>;
@group(0) @binding(5) var<storage, read_write> rng_state: array<u32>;
@group(0) @binding(6) var<uniform> params: SamplerParams;

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
      // Softcap bounds logits to [0, 23]; subtracting 23 keeps exp finite.
      let scaled = (candidate + 5.0f) / 7.5f;
      let exponential = exp(-abs(scaled));
      let sigmoid = select(exponential / (1.0f + exponential),
        1.0f / (1.0f + exponential), scaled >= 0.0f);
      value = exp((23.0f * sigmoid - 23.0f) / params.temperature);
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
  let result = reduce_values[0];
  workgroupBarrier();
  return result;
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
