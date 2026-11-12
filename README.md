# Weight-space programming for college advising

A replication and extension of [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (Takehi, Teoh, Zhang & Khattab, MIT CSAIL) in a domain where the failures are judgment rather than arithmetic.

Five frontier models were each given direct read/write access to the weights of Qwen3.5-0.8B, 40 forward passes, and a brief describing five advising failures. No gradients, no training loop. They read the numbers, formed hypotheses about which numbers do what, changed some, and measured.

## Result

All figures below are from **141 replacement items written after a leak in the harness was found, which no editor ever saw**. See [the correction](#the-correction) — it is the most transferable part of this work.

| arm | weights changed | mean | fabrication | severe errors | editor spend |
|---|---|---|---|---|---|
| Qwen3.5-0.8B as is | — | 50.7% | 39.3% | 28.9% | — |
| **Sonnet 5.5 edit** | **18,432** | 60.9% | **92.9%** | **9.1%** | $87.78 |
| Opus 5.5 edit | 6,144 | 56.4% | 89.3% | 18.9% | $106.39 |
| advising context only | 0 | **63.4%** | 82.1% | 18.9% | $0 |
| Astra at `xhigh` | 3,072 | 53.7% | 64.3% | 20.4% | $12.22 |
| Astra at `high` | 3,072 | 54.3% | 53.6% | 23.2% | $13.70 |
| Astra, 80 passes | identical md5 | 54.3% | 53.6% | 23.2% | $23.87 |
| Fable 5.1 edit | 60,000,000 | 55.1% | 71.4% | 23.5% | $110.73 |
| Kimi K3 (reverted all) | 0 | 50.9% | 39.3% | 30.4% | $18.48 |

n = 2 per item, identical seeds for base and treatment, paired standard errors.

Four findings worth the reading time:

1. **The smallest edit won.** 18,432 weights — 0.0024% of the model — beat an edit three thousand times larger on every measure. For scale, the LoRA fine-tune of the same model changes **255,965,539** weights, or 34% of it.
2. **A good prompt beats every edit on the mean, and loses on the floor.** Context injection tops the average at 63.4; editing is the only lever that brought severe errors into single digits. The two are not substitutes, and the mean hides it.
3. **Four of five editors independently found the same broken component** — one neuron in the last MLP whose gate reads "asks for a specific the model cannot know" with the wrong sign, so it goes quiet exactly when the model should defer. Flipping 2,048 weights moves P("I") from 0.05 to 0.85.
4. **The editor that declined to edit made the model worse.** Kimi read the failures as missing knowledge, reverted both of its edits, and is the only arm whose severe-error rate rose.

## The correction

Partway through, four of the five editors could read the evaluation answer key through a bug in this harness: the forward-pass server globbed the directory the held-out sets lived in. Sweeping the tree properly found **406 paths** carrying held-out content by three independent routes — the one that mattered was `work/eval_*/rows_heldout_*.jsonl`, written by the harness into the editor's own working directory, carrying both the items and the grader patterns.

Every frozen edit was re-measured on replacement items. **The leak did not inflate the headline** — one editor's fabrication gain came back at +32.1 against the +32.2 it scored with the key in front of it. **What the leak did was invert the ranking.** On the compromised set the largest edit appeared to win on severe errors, 10.9% against 14.4%; clean, that reverses to 9.1% against 23.5%. The compromised set had a 15.3% floor and could not separate them.

The lesson, which is cheaper to borrow than to earn: *a contaminated evaluation can reproduce your effect faithfully and still get your ordering backwards.* "It replicated" is not evidence a leak was harmless.

## Layout

```
harness/     kernel.py (weight read/write), server.py (forward-pass server),
             editor_api.py (the editor loop), eval_heldout.py, run_compare.sh
runners/     the pod and laptop launchers, including the sealing runner that
             moves every held-out path out of an editor's tree before it starts
tasks/       the briefs each editor was given
contexts/    the 1.3k-token advising prompt used by the context arms
finetune/    LoRA training, curation, and the merged-model -> per-tensor converter
results/     clean/    one summary.json per arm, replacement set
             original/ one summary.json per arm, compromised set — kept apart
             on purpose so the two can never be read in the same column
editors/     each editor's own written conclusion and arm metadata
docs/        the memo, the blog post, and the full clean-results record
index.html   the published comparison page
```

## Deliberately not here

- **The graded evaluation items**, because publishing a benchmark burns it.
- **The evaluation-set generators.** The blog post says these are included; they are not, and that is a deliberate departure. `make_eval_v2.py` and its siblings embed both the item templates and the grader patterns, so shipping them publishes the benchmark by another route. Open an issue if you want them for replication and they can be shared directly.
- **The editor transcripts**, which contain held-out items verbatim.
- **The edits as weight files.** The small arms are a few hundred KB as sparse diffs and will be added; the Fable arm is 60M changed weights and the LoRA 256M, neither of which belongs in git.

## Reproducing

The harness expects a Hugging Face folder of Qwen3.5-0.8B one level above `harness/`. `kernel.py` reads and writes tensors by name with no framework dependency beyond numpy — it parses safetensors directly — so analysis runs anywhere, while generation needs a GPU or an M-series Mac.

Nothing here will run an editor without API keys, which are read from a `.env` that is not in this repo and never should be.
