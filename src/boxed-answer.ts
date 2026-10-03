export type BoxedAnswer = {
  readonly content: string;
  readonly commandStart: number;
  readonly closeBrace: number;
};

type MathDelimiter = "\\[" | "\\(" | "$$" | "$";

/** Finds the last complete, non-empty `\\boxed { ... }` expression. */
export function findLastBoxedAnswer(source: string): BoxedAnswer | null {
  const command = "\\boxed";
  let result: BoxedAnswer | null = null;
  let cursor = 0;

  while (cursor < source.length) {
    const commandStart = source.indexOf(command, cursor);
    if (commandStart < 0) break;

    let openBrace = commandStart + command.length;
    while (/\s/.test(source[openBrace] ?? "")) openBrace += 1;
    if (source[openBrace] !== "{") {
      cursor = Math.max(openBrace, commandStart + command.length);
      continue;
    }

    let depth = 1;
    let closeBrace = -1;
    for (let index = openBrace + 1; index < source.length; index += 1) {
      if (source[index] === "\\") {
        index += 1;
        continue;
      }
      if (source[index] === "{") depth += 1;
      if (source[index] === "}") depth -= 1;
      if (depth === 0) {
        closeBrace = index;
        break;
      }
    }

    if (closeBrace < 0) break;
    const content = source.slice(openBrace + 1, closeBrace);
    if (content.trim()) {
      result = { content, commandStart, closeBrace };
    }
    cursor = closeBrace + 1;
  }

  return result;
}

/**
 * Returns true only after the answer box and any math delimiter enclosing it
 * have both completed. This keeps the closing `$`, `$$`, `\\)`, or `\\]` in
 * the streamed answer instead of stopping on the box's closing brace.
 */
export function hasCompleteFinalBox(source: string): boolean {
  const thinkStart = source.indexOf("<think>");
  const thinkEnd = thinkStart >= 0
    ? source.indexOf("</think>", thinkStart + "<think>".length)
    : -1;
  if (thinkStart >= 0 && thinkEnd < 0) return false;

  const answerStart = thinkEnd >= 0 ? thinkEnd + "</think>".length : 0;
  const answer = source.slice(answerStart);
  const box = findLastBoxedAnswer(answer);
  if (!box) return false;

  const delimiter = activeMathDelimiter(answer, box.commandStart);
  return delimiter === null || hasClosingDelimiter(answer, box.closeBrace + 1, delimiter);
}

function activeMathDelimiter(source: string, end: number): MathDelimiter | null {
  let active: MathDelimiter | null = null;
  for (let index = 0; index < end; index += 1) {
    if (source.startsWith("\\[", index)) {
      if (active === null) active = "\\[";
      index += 1;
      continue;
    }
    if (source.startsWith("\\]", index)) {
      if (active === "\\[") active = null;
      index += 1;
      continue;
    }
    if (source.startsWith("\\(", index)) {
      if (active === null) active = "\\(";
      index += 1;
      continue;
    }
    if (source.startsWith("\\)", index)) {
      if (active === "\\(") active = null;
      index += 1;
      continue;
    }
    if (source[index] !== "$" || isEscaped(source, index)) continue;
    const delimiter: MathDelimiter = source[index + 1] === "$" ? "$$" : "$";
    if (delimiter === "$$") index += 1;
    if (active === delimiter) active = null;
    else if (active === null) active = delimiter;
  }
  return active;
}

function hasClosingDelimiter(
  source: string,
  start: number,
  delimiter: MathDelimiter,
): boolean {
  const closer = delimiter === "\\[" ? "\\]" : delimiter === "\\(" ? "\\)" : delimiter;
  let cursor = start;
  while (cursor < source.length) {
    const found = source.indexOf(closer, cursor);
    if (found < 0) return false;
    if (!closer.startsWith("$") || !isEscaped(source, found)) return true;
    cursor = found + closer.length;
  }
  return false;
}

function isEscaped(source: string, index: number): boolean {
  let slashes = 0;
  for (let cursor = index - 1; cursor >= 0 && source[cursor] === "\\"; cursor -= 1) {
    slashes += 1;
  }
  return slashes % 2 === 1;
}
