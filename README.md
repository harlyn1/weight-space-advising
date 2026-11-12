# Weight-space programming for college advising

A replication and extension of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang & Khattab, MIT CSAIL) in a domain where the failures are judgment rather
than arithmetic.

Five frontier models were each given direct read/write access to the weights of Qwen3.5-0.8B,
40 forward passes, and a brief describing five advising failures. No gradients, no training
loop. They read the numbers, formed hypotheses about which numbers do what, changed some, and
measured.

This repo follows the layout of the original paper's log release: one folder per experiment,
each with its own README mapping runs to reported numbers and to the eval file each number
came from.

## Two sets, and never mix them

| | items | base mean / fabrication / severe |
|---|---|---|
| **clean** | 141 replacement items written after the leak was found; no editor ever saw them | 50.7 / 39.3 / 28.9% |
| **original** | the compromised held-out set — four editors could read graded answers from it | 62.5 / 57.1 / 15.3% |

Eval files are suffixed `.clean.json` or `.original.json`. The replacement set is markedly
harder, so a number from one set can never sit in the same column as a number from the other.

## Headline, clean set

| arm | lever | weights changed | mean | fabrication | severe |
|---|---|---|---|---|---|
| base | — | — | 50.7 | 39.3 | 28.9% |
| LoRA, re-weighted curriculum | fine tune | 257,288,555 | **65.5** | 92.9 | **7.0%** |
| Sonnet 5.5 edit | edit | **18,432** | 60.9 | 92.9 | 9.1% |
| Sonnet edit + context | both | 18,432 | **67.9** | 92.9 | 11.9% |
| advising context | prompt | 0 | 63.4 | 82.1 | 18.9% |
| Fable, built blind | edit | — | 58.8 | 89.3 | 14.7% |
| LoRA, original curriculum | fine tune | 255,965,539 | 61.8 | 60.7 | 21.9% |

Four results worth the reading time:

1. **Four of five editors independently found the same broken component** — one neuron whose
   gate reads "asks for a specific the model cannot know" with the wrong sign (exp8).
2. **A contaminated eval can reproduce your effect and still invert your ranking** (exp7). The
   fabrication gain replicated to a tenth of a point; the ordering on dangerous answers reversed.
3. **An editor that cannot see the test set builds a better edit** — 89.3 fabrication against
   71.4, at less than half the cost (exp7).
4. **Fine tuning's failure was aim, not budget** (exp5). Re-weighting a 225-example curriculum
   from 7% to 40% refusal, at no extra cost, turns the worst lever into the best one.

The original write-up claimed the smallest edit won outright. It does not survive a competently
aimed fine-tune. What survives is that the edit matched that result with 18,432 weights against
257 million, and with no training data at all.

## Layout

```
harness/                 kernel (weight read/write), forward-pass server, editor loop,
                         evaluation, arm comparison
runners/                 pod and laptop launchers
exp1_baseline/           the unedited model, and a frontier ceiling
exp2_editors/            five editors at 40 passes, with each one's written conclusion
exp3_reasoning_effort/   pass budget against reasoning effort
exp4_context/            context injection, alone and on top of each edit
exp5_finetune/           three LoRA runs that differ only in curriculum
exp6_scale_4b/           does the recipe transfer to a 4B model
exp7_leak_and_correction/ the 406 paths, the replacement set, the blind rebuild
exp8_mechanism/          the neuron four editors found
docs/                    memo, blog post, full clean-results record
index.html               the published comparison page
```

## Deliberately not here

- **The graded evaluation items**, because publishing a benchmark burns it.
- **The evaluation-set generators.** The write-up says these are included; they are not, and
  that is a deliberate departure. They embed both the item templates and the grader patterns,
  so publishing them would publish the benchmark by another route. Open an issue if you want
  them for replication.
- **The editor transcripts**, which contain held-out items verbatim.
- **Weight files.** Sparse diffs for the small arms will be added; Fable's is 60M changed
  weights and the LoRAs 256M, neither of which belongs in git.

## Reproducing

`harness/kernel.py` reads and writes tensors by name and parses safetensors directly, so
analysis needs only numpy. Generation needs a GPU or an M-series Mac. The harness expects a
Hugging Face folder of Qwen3.5-0.8B one level above it.

Nothing here runs an editor without API keys, which are read from a `.env` that is not in this
repo and never should be.
