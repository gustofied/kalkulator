# Kalkulator

Browser-only playground for Paradigma’s Limite 1B Violetto, running locally in your browser. Runs with JAX.js and WebGPU.

[Open Kalkulator](https://kalkulator.adamsioud.com)

## Browser requirements

- A current desktop browser with WebGPU support
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
