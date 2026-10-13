// Stop an uninterrupted token cycle, not ordinary repeated words or equations.
// At least 128 tokens and four complete copies must agree exactly.
export function hasRepeatingSuffix(tokens: readonly number[]): boolean {
  if (tokens.length < 128 || tokens.length % 16 !== 0) return false;
  for (let period = 1; period <= 64; period++) {
    const span = Math.max(128, period * 4);
    if (tokens.length < span) continue;
    let matches = true;
    for (let offset = period; offset < span; offset++) {
      if (tokens[tokens.length - 1 - offset] !== tokens[tokens.length - 1 - offset + period]) {
        matches = false;
        break;
      }
    }
    if (matches) return true;
  }
  return false;
}
