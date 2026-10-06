// The single production model and decoding policy. No runtime overrides.
export const LIMITE_MANIFEST_URL =
  "https://huggingface.co/gustofied/kalkulator/resolve/f66f2e77b2bfd4e8aa80e0770943c3c0d3414f18/webgpu-q4-v2/manifest.json";
export const LIMITE_TOKENIZER_URL =
  "https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/tokenizer.json";

export const LIMITE_TOKENIZER_VOCAB_SIZE = 151_667;
export const LIMITE_EOS_TOKEN = 151_643;
export const LIMITE_CONTEXT_TOKENS = 131_072;
export const LIMITE_MAX_OUTPUT_TOKENS = 126_976;
export const LIMITE_TEMPERATURE = 0.6;
export const LIMITE_TOP_P = 0.95;
export const LIMITE_DECODE_BATCH_SIZE = 4;
// Recover stalled execution, not a healthy calculation that is still progressing.
export const LIMITE_STALL_TIMEOUT_MS = 150_000;
