/** Recover source math from rich clipboard content without inserting its HTML. */
export function mathFromClipboard(html: string): string | null {
  if (!html || html.length > 1_000_000) return null;
  // Template contents are inert: copied scripts, images and frames never run or
  // enter the document. Only plain text leaves this function.
  const template = document.createElement("template");
  template.innerHTML = html;
  const content = template.content;
  const equations = content.querySelectorAll(".katex");
  if (!equations.length) return null;

  for (const equation of equations) {
    const latex = equation
      .querySelector('annotation[encoding="application/x-tex"]')
      ?.textContent?.trim();
    // Do not guess from visually positioned glyphs if source math is missing.
    if (!latex) return null;
    const suffix = equation.matches("div, p") ? "\n" : "";
    equation.replaceWith(document.createTextNode(`\\(${latex}\\)${suffix}`));
  }
  content
    .querySelectorAll("script, style, template, iframe, object, embed, link, meta, img, svg")
    .forEach((node) => node.remove());
  content.querySelectorAll("br")
    .forEach((node) => node.replaceWith(document.createTextNode("\n")));
  content.querySelectorAll("p, div, li, blockquote, h1, h2, h3, h4, h5, h6, tr")
    .forEach((node) => node.append("\n"));
  return content.textContent
    ?.replace(/[^\S\n]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim() || null;
}
