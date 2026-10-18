import { jit, nn, numpy as np, random } from "@jax-js/jax";
import { LIMITE_TEMPERATURE, LIMITE_TOKENIZER_VOCAB_SIZE, LIMITE_TOP_P } from "./limite-config";

function sumVocabulary(values: np.Array): np.Array {
  const padding = (256 - values.size % 256) % 256;
  return np.pad(values, [[0, padding]]).reshape([-1, 256]).sum(-1).sum();
}

/** Full-vocabulary nucleus sampling, with no top-k approximation. */
export const sampleToken = jit((rawLogits: np.Array, key: np.Array): [np.Array, np.Array] => {
  const validLogits = rawLogits.slice([0, LIMITE_TOKENIZER_VOCAB_SIZE]);
  const invalid = sumVocabulary(np.isfinite(validLogits.ref).astype(np.uint32)).notEqual(LIMITE_TOKENIZER_VOCAB_SIZE);
  const logits = nn.sigmoid(validLogits.add(5).div(7.5))
    .mul(23 / LIMITE_TEMPERATURE);
  const order = np.argsort(logits.ref.neg());
  // The sigmoid softcap bounds the maximum, so normalization needs no max pass.
  const masses = np.exp(np.take(logits, order.ref).sub(23 / LIMITE_TEMPERATURE));
  const probabilities = masses.ref.div(sumVocabulary(masses));
  const cumulative = np.cumsum(probabilities.ref);
  // Include the token that crosses the threshold, as in standard top-p sampling.
  // Use signed indices so subtracting one cannot saturate an unsigned cast.
  const lastIndex = sumVocabulary(cumulative.ref.sub(probabilities)
    .less(LIMITE_TOP_P).astype(np.int32)).sub(1);
  const keys = random.split(key);
  const uniform = random.uniform(keys.ref.slice(0));
  // The retained prefix already has its CDF; no masked second scan is needed.
  const threshold = uniform.mul(np.take(cumulative.ref, lastIndex.ref));
  const index = np.minimum(sumVocabulary(cumulative.lessEqual(threshold).astype(np.int32)), lastIndex);
  return [np.where(invalid, LIMITE_TOKENIZER_VOCAB_SIZE, np.take(order, index)), keys.slice(1)];
});
