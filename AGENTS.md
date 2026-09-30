# Kalkulator

## Purpose

Run Paradigma's Violetto Limite 1B math model entirely in the browser with WebGPU.

## Commands

- Install: `npm ci`
- Develop: `npm run dev`
- Verify: `npm run build`

## Constraints

- Keep the public app static and browser-only; do not add a backend without an explicit decision.
- Preserve the small, editorial black/violet interface.
- Do not commit model weights or add a prose README.
- Treat `src/model.ts` as the JAX.js reference implementation of Violetto.
- Verify inference in a real WebGPU browser after changing model execution.
- Keep the GitHub Pages deployment working.

## Project context

Read `docs/PROJECT.md` before changing the runtime, model format, caching, or deployment.

