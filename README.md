<h1 align="center">Kalkulator</h1>

<p align="center">
  <img src="public/gothic-k-scholar.png" alt="Kalkulator engraved initial" width="96" height="96">
</p>

<p align="center">
  <a href="https://gustofied.github.io/kalkulator/">Open Kalkulator</a>
</p>

- [Paradigma’s Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) for maths, running entirely in your browser.
- [JAX.js](https://github.com/ekzhang/jax-js) and WebGPU, with no inference server.
- 556 MiB of quantised weights with local browser caching.

## Example

Add $`\frac{7}{12}`$ and $`\frac{5}{18}`$.

```math
\frac{21}{36} + \frac{10}{36} = \boxed{\frac{31}{36}}
```

## Run locally

```sh
npm ci
npm run dev
```
