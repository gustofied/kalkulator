# Kalkulator

[Paradigma’s Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) for mathematics, running in your browser with [JAX.js](https://github.com/ekzhang/jax-js) and WebGPU.

[Open Kalkulator](https://gustofied.github.io/kalkulator/)

## Browser requirements

- A current browser with WebGPU support
- Enough memory to load the 1.93 GiB model

## Local setup

Requires Node.js 22 or newer.

```sh
git clone https://github.com/gustofied/kalkulator.git
cd kalkulator
npm ci
npm run dev
```

Build with `npm run build`.
