# Browser verification, 2026-10-06

Report-aligned runtime: unchanged 555.87 MiB Q4 artifact, 131,072-token context, 126,976-token maximum output, packed FP16 attention cache with FP32 computation. One engine, no model selector, runtime probes, answer forcing, or automatic retries. EOS replaces stop-on-box; inactivity recovery replaces the two-minute solve cutoff.

Verified in the real WebGPU browser at `http://127.0.0.1:5185/`:

- Final end-to-end smoke: `17 + 25` returned boxed `42`, stopped on EOS, and re-enabled the form. 430 tokens, 56.29 decode tok/s, 523.7 ms prefill, 8.147 s total. This is a spot measurement, not evidence of a speedup over a different sampled response.
- Packed attention agreed with an independent CPU calculation within `1e-7` absolute error on synthetic global positions 128, 4096, and 131071, and local-ring position 1031, including XSA and output gating. This verifies the checked attention arithmetic, not full-model parity.
- Actual engine prefill of 4,094 tokens followed by a four-token decode crossed the 4K allocation boundary. Reset, cache shrink, and another decode succeeded with valid token IDs and no WebGPU validation errors.
- `npm run build` and `git diff --check` passed. Temporary verification pages were removed; none are shipped.

The larger budget removes a known truncation mismatch; it does not establish hard-question reliability or official-weight numerical parity. Full-model generation through 128K has not been measured. Initial attention-cache allocation is about 84 MiB; the full 128K cache is about 1,572 MiB, excluding weights and temporary growth allocations.

## Bounded blog checks, 2026-10-06

Two problems from [Paradigma's release post](https://paradigma.inc/blog/limite-1b-violetto/) were entered through the actual notebook UI, using plain-text mathematical notation without answer hints. One attempt per question, with a five-minute observation window declared in advance. The Q4 artifact, prompt template, sampling, and production generation limits were unchanged. There were no retries or selected seeds. Each unfinished run was stopped by reloading the page, not by a runtime time limit.

| Question | Expected answer | Observed result | Observed elapsed | Last logged tokens | Last logged average decode tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| Two equal minima | `240` | Still generating; no final answer | 301 s | 9,986 | 34.54 |
| Jessica's brick wall | `3^2025` | Still generating; no final answer | 302 s | 6,918 | 23.73 |

Token counts are lower bounds from the last periodic console sample, not exact counts at cancellation. The associated speeds are those samples' cumulative averages. They are not controlled speedup comparisons; the second run was slower even at shorter context. No WebGPU errors were observed. Reloads used the cached model and returned to the ready state.

**0 of 2 attempts produced a completed answer within five minutes.** These are incomplete responses, not two graded wrong answers, and they do not establish whether the model would succeed with more time. The fast-answer goal remains unmet. No AIME benchmark, full reference reproduction, or additional optimization loop was run. No temporary scripts, pages, or model copies were created for these checks.

# Previous candidate, 2026-10-05

Production candidate: the existing 555.87 MiB Q4 artifact, full-vocabulary top-p 0.95 at temperature 0.6, 16,384-token context, 120-second solve budget. One inference worker and one generation stream. No seed selection, retries, answer hints, or second model.

Checks run in the real WebGPU in-app browser at `http://127.0.0.1:5185/`. These are individual samples, not success-rate estimates or parity claims. Blog questions are transcribed into plain text with mathematical meaning preserved; rendered-math duplication is removed.

| Check | Result |
| --- | --- |
| `17 + 25` | Correct boxed `42`; 277 tokens, 53.63 tok/s, 5.80 seconds. |
| GPU sampler: padded rows with artificially dominant logits | Excluded; valid winner `42` sampled. |
| GPU sampler: uniform full vocabulary | Sampled ID `76316`, within the 95% nucleus and beyond the old top-50 limit. |
| GPU sampler: all invalid logits | Ends with EOS `151643`. |
| Answer boundaries | Ten checks passed, including nested boxes, unclosed math delimiters, and a box inside unfinished thinking. |

## Blog-question checks

One attempt per question, with the same two-minute production budget. These are completion checks, not evidence of upstream numerical parity.

| Question | Expected answer | Observed result | Tokens | Decode tok/s | Total seconds |
| --- | --- | --- | ---: | ---: | ---: |
| Jessica's brick wall | `3^2025` | Time limit, no final answer | 3,997 | 33.95 | 120.01 |
| Three minima | `240` | Time limit, no final answer | 4,033 | 33.98 | 120.10 |
| Triangle centers | `2*sqrt(435)/3` | Time limit, no final answer | 4,037 | 33.95 | 120.02 |
| Calvin's board | `1080` | Time limit, no final answer | 3,933 | 33.59 | 120.13 |

Each completed bounded run returned control to the form; successive problems started without reloading or downloading the model. **0 of 4 blog examples reached a final answer within the useful waiting-time budget in this check.** The original answer-quality goal remains unmet. The sampler correction, larger context, cache verification, and explicit completion states improve the implementation contract; they do not demonstrate mathematical reliability or a successful speed optimization.

## Historical result, not current acceptance

On 2026-10-02 an experimental mixed-precision V12 build returned the triangle-center answer `2*sqrt(435)/3` after 13,697 tokens. Two fresh workers produced the same output hash, at 25.49 and 22.59 tok/s. That artifact and decoding policy differ from today's Q4 production candidate. It does not establish acceptance of the current build or the other three blog problems.

Source questions: <https://paradigma.inc/blog/limite-1b-violetto/>.
