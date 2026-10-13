export type BoxedAnswer = {
  readonly content: string;
  readonly commandStart: number;
  readonly closeBrace: number;
};

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
 * Validate only after EOS. The box is rendered independently, so surrounding
 * prose (including currency symbols) must not affect its math delimiters.
 */
export function hasCompleteFinalBox(source: string): boolean {
  const thinkStart = source.lastIndexOf("<think>");
  const thinkEnd = source.lastIndexOf("</think>");
  if (thinkStart > thinkEnd) return false;

  const answerStart = thinkEnd >= 0 ? thinkEnd + "</think>".length : 0;
  const answer = source.slice(answerStart);
  return findLastBoxedAnswer(answer) !== null;
}

/** Keep a final boxed equation together instead of leaving its left half behind. */
export function finalAnswerStart(source: string, box: BoxedAnswer): number {
  const prefix = source.slice(0, box.commandStart);
  let start = box.commandStart;
  for (const [open, close] of [["\\[", "\\]"], ["\\(", "\\)"]] as const) {
    const opening = prefix.lastIndexOf(open);
    if (opening > prefix.lastIndexOf(close)) start = Math.min(start, opening);
  }
  const displays = [...prefix.matchAll(/\$\$/g)];
  if (displays.length % 2) start = Math.min(start, displays.at(-1)!.index);
  return start;
}
