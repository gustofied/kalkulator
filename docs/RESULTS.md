# Browser verification, 2026-10-05

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
