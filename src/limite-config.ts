// The single production model and decoding policy. No runtime overrides.
export const LIMITE_MANIFEST_URL =
  "https://huggingface.co/gustofied/kalkulator/resolve/ebc9ad0db914085509ba5d8f39f100d1a09c18a8/webgpu-q4-v3/manifest.json";
export const LIMITE_TOKENIZER_URL =
  "https://huggingface.co/paradigma-inc/limite-1b-violetto/resolve/b1f3d572ccacb6919f4d64c321b70ba034ddaef2/tokenizer.json";
export const LIMITE_MANIFEST_SHA256 =
  "54cb625e041037cb6191864ecf555cd1039666cd20e06f4f994bd1877c7811c1";
export const LIMITE_TOKENIZER_SHA256 =
  "296e081e2f5ecf9d87814aa9b0f4b12d670ed2b2e2be6c84e01a9466c953afb7";

export const LIMITE_TOKENIZER_VOCAB_SIZE = 151_667;
export const LIMITE_EOS_TOKEN = 151_643;
export const LIMITE_CONTEXT_TOKENS = 131_072;
export const LIMITE_MAX_OUTPUT_TOKENS = 126_976;
export const LIMITE_TEMPERATURE = 0.6;
export const LIMITE_TOP_P = 0.95;
export const LIMITE_DECODE_BATCH_SIZE = 4;
// A notebook calculation must finish or return control, even if tokens keep arriving.
export const LIMITE_SOLVE_TIMEOUT_MS = 180_000;
// Recover stalled execution, not a healthy calculation that is still progressing.
export const LIMITE_STALL_TIMEOUT_MS = 150_000;
