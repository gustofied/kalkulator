import { jit, numpy as np } from "@jax-js/jax";

export type Q4Linear = { codes: np.Array; scales: np.Array };

/** Signed block-32 Q4, expressed entirely as JAX.js array operations. */
export function unpackQ4({ codes, scales }: Q4Linear): np.Array {
  const [rows, words] = codes.shape;
  const shifts = np.arange(8, undefined, undefined, { dtype: np.uint32 }).mul(4);
  const values = np.bitwiseAnd(
    np.rightShift(codes.reshape([rows, words, 1]), shifts), 15,
  ).astype(np.float32).sub(8).reshape([rows, words / 4, 32]);
  return values.mul(scales.astype(np.float32).reshape([rows, words / 4, 1]))
    .reshape([rows, words * 8]);
}

export function q4Linear(weight: Q4Linear, x: np.Array): np.Array {
  if (x.shape[0] !== 1) return np.dot(x, unpackQ4(weight).transpose());
  const rows = weight.codes.shape[0];
  // JIT fuses Q4 unpacking, block scales and the row reduction into one kernel;
  // decode does not materialize the unpacked weight matrix or block sums.
  return unpackQ4(weight).mul(x).sum(-1).reshape([1, rows]);
}

export const q4Embedding = jit((weight: Q4Linear, ids: np.Array): np.Array =>
  unpackQ4({ codes: weight.codes.slice(ids.ref), scales: weight.scales.slice(ids) }),
);
