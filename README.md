# Kalkulator

[Paradigma’s Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) for mathematics, running privately in your browser with a model-specific WebGPU engine. The original reference port uses [JAX.js](https://github.com/ekzhang/jax-js).

[Open Kalkulator](https://gustofied.github.io/kalkulator/)

## Browser requirements

- A current browser with WebGPU support
- Enough memory and site storage for the 556 MiB model artifact

## Local setup

Requires Node.js 22 or newer.

```sh
git clone https://github.com/gustofied/kalkulator.git
cd kalkulator
npm ci
npm run dev
```

Build with `npm run build`.
