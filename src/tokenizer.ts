import { tokenizers } from "@jax-js/loaders";
import { LIMITE_EOS_TOKEN } from "./limite-config";

type AddedToken = { id: number; content: string; special: boolean };
type TokenizerData = {
  added_tokens: AddedToken[];
  pre_tokenizer: {
    type: "Sequence";
    pretokenizers: [
      { type: "Split"; pattern: { Regex: string } },
      { type: "ByteLevel" },
    ];
  };
  model: { type: "BPE"; vocab: Record<string, number> };
};

export class ViolettoTokenizer {
  readonly eosToken = LIMITE_EOS_TOKEN;
  readonly #encoding: tokenizers.BpeEncoding;

  constructor(data: TokenizerData) {
    const added = new Map(data.added_tokens.map((token) => [token.id, token]));
    const special: Record<string, number> = {};
    for (const token of data.added_tokens) {
      if (token.special || !(token.content in data.model.vocab)) {
        special[token.content] = token.id;
      }
    }

    const byteDecoder = createByteDecoder();
    const encoder = new Map<string, number>();
    for (const [piece, id] of Object.entries(data.model.vocab)) {
      if (added.get(id)?.special) continue;
      encoder.set(decodeByteLevelPiece(piece, byteDecoder), id);
    }

    const split = data.pre_tokenizer.pretokenizers[0];
    const pattern = split.pattern.Regex.replace(/^\(\?i:([^)]*)\)/, "(?:$1)");
    this.#encoding = new tokenizers.BpeEncoding(
      encoder,
      special,
      new RegExp(pattern, "giu"),
    );
  }

  static fromBinary(data: Uint8Array): ViolettoTokenizer {
    return new ViolettoTokenizer(JSON.parse(new TextDecoder().decode(data)));
  }

  encode(text: string): number[] {
    return this.#encoding.encodeWithSpecialTokens(text.normalize("NFC"));
  }

  createDecoder(): ViolettoStreamDecoder {
    return new ViolettoStreamDecoder(this.#encoding);
  }
}

export class ViolettoStreamDecoder {
  readonly #encoding: tokenizers.BpeEncoding;
  readonly #utf8 = new TextDecoder();
  #text = "";

  constructor(encoding: tokenizers.BpeEncoding) {
    this.#encoding = encoding;
  }

  push(token: number): string {
    this.#text += this.#utf8.decode(this.#encoding.decodeBytes([token]), {
      stream: true,
    });
    return this.#text;
  }

  finish(): string {
    this.#text += this.#utf8.decode();
    return this.#text;
  }
}

function createByteDecoder(): Map<string, number> {
  const bytes: number[] = [];
  for (let i = 33; i <= 126; i++) bytes.push(i);
  for (let i = 161; i <= 172; i++) bytes.push(i);
  for (let i = 174; i <= 255; i++) bytes.push(i);
  const chars = [...bytes];
  let extra = 0;
  for (let byte = 0; byte < 256; byte++) {
    if (bytes.includes(byte)) continue;
    bytes.push(byte);
    chars.push(256 + extra++);
  }
  return new Map(chars.map((char, i) => [String.fromCodePoint(char), bytes[i]]));
}

function decodeByteLevelPiece(piece: string, decoder: Map<string, number>): string {
  let hex = "";
  for (const char of piece) {
    const byte = decoder.get(char);
    if (byte === undefined) throw new Error("Invalid tokenizer byte");
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}
