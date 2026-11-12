<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp7 — the leak, the replacement set, and the blind rebuild

Partway through, four of the five editors could read the evaluation answer key.

**The bug.** The forward-pass server loaded every file in `practice/`, and the held-out sets
lived there. Sweeping the tree properly turned up **406 paths** carrying held-out content by
three independent routes: `exports/`, `gpu/box_sync/work/`, and — the one that mattered —
`work/eval_*/rows_heldout_*.jsonl`, written by the harness into the editor's own working
directory, carrying both the items and the grader patterns. Closing the first route alone would
not have made a later run clean.

**Exposure, by save timing.** Graded-answer counts understate it. Fable read the held-out files
at turn 47 and Opus at turn 87 — in both cases *before either wrote a single weight* — and all
16 of Fable's and all 10 of Opus's saves came after. Astra made 3 of 9 saves after exposure, and
its submitted weights are byte-identical to a pre-exposure snapshot. Kimi made 7 of 10 but
reverted everything. **Sonnet had zero exposure across 383 turns.**

**The fix.** `seal/fable_08b_clean.sh` moves every matching path out of the editor's tree before
the editor starts and aborts if any remains. It matches by pattern, and restricts the
`*heldout*` file match to `*.jsonl` so the harness's own `eval_heldout.py` is not swept away.
`seal/remeasure_v2.sh` re-measures frozen edits against the replacement set via `WSP_EVAL_DIR`,
leaving the original `eval/heldout/` untouched so old numbers stay comparable with each other.

**What re-measuring showed.** The headline replicated: Fable's fabrication gain came back at
+32.1 against the +32.2 it scored with the key in front of it. **What the leak had distorted was
the ranking.** On the compromised set the largest edit appeared to win on dangerous answers,
10.9% against 14.4%; clean, that reverses to 9.1% against 23.5%. The compromised set had a 15.3%
floor and could not separate them.

> A contaminated evaluation can reproduce your effect faithfully and still get your ordering
> backwards. "It replicated" is not evidence that a leak was harmless.

**The blind rebuild.** Re-running Fable with all 406 paths sealed produced a *better* edit than
the same editor with access: 89.3 fabrication against 71.4, and 14.7% dangerous answers against
23.5%, for $49.51 against $110.73. Seeing the test set let it tune against specific items instead
of finding the general mechanism.

| run | mean | fabrication | severe | eval file |
|---|---|---|---|---|
| grader validation | 47.6 | 28.6 | 38.4% | `eval/validate.clean.json` |
| Fable, built blind | 58.8 | 89.3 | 14.7% | (on pod) |

> **Pending fetch.** Some eval files listed below were produced on the GPU pod and have not
> been copied into this repo yet. They are marked `(on pod)`. The pod's disk is ephemeral;
> `runners/pull_results.command` retrieves them.
