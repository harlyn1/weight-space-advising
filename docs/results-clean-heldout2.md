# Clean results — heldout2 re-measurement

Nov 11, 2026 · supersedes `results-v1.md` for every arm listed here

## What this set is

141 items generated from the same six family generators as the original held-out set, written **after** the leak was found, and never present in any editor's working tree. Staged at `/opt/wsp_v2/eval` on the pod and read through `WSP_EVAL_DIR`, so the original `eval/heldout/` was never overwritten and the old numbers stay comparable with each other.

Graders were validated first: the unedited model was run against the set at n=1 and the per-family failures inspected by hand before any treatment arm was measured. Arms are named `v2_*` and take deltas against `v2_base` via `WSP_BASE_ARM`.

**The replacement set is harder than the original.** Base mean 50.7 against 62.5; base severe-error rate 28.9% against 15.3%. No figure here may be placed beside an old-set figure.

## Results

n = 2 per item, same seeds for base and treatment, paired SE. Ordered by severe-error rate.

| arm | mean | Δ | fabrication | Δ | severe | Δ | held-out exposure |
|---|---|---|---|---|---|---|---|
| base | 50.7 | — | 39.3 | — | 28.9% | — | no editor |
| **sonnet** | 60.9 | +10.2 | **92.9** | +53.6 | **9.1%** | −19.8 | **none: 0 of 13 saves after exposure** |
| opus | 56.4 | +5.7 | 89.3 | +50.0 | 18.9% | −10.0 | first read turn 87; 10 of 10 saves after |
| context | **63.4** | +12.7 | 82.1 | +42.8 | 18.9% | −10.0 | no editor |
| astra_xhigh | 53.7 | +3.0 | 64.3 | +25.0 | 20.4% | −8.5 | exposed, but final = pre-exposure snapshot |
| astra | 54.3 | +3.6 | 53.6 | +14.3 | 23.2% | −5.7 | 3 of 9 saves after exposure |
| astra80 | 54.3 | +3.6 | 53.6 | +14.3 | 23.2% | −5.7 | byte-identical to astra |
| fable | 55.1 | +4.4 | 71.4 | +32.1 | 23.5% | −5.4 | first read turn 47; 16 of 16 saves after |
| kimi | 50.9 | +0.2 | 39.3 | 0.0 | 30.4% | +1.5 | 7 of 10 saves after exposure |

## What changed against the compromised set

**The headline replicated.** Fable's fabrication gain came back at **+32.1 against its leaked +32.2**. Opus replicated at +50.0. The mechanism generalizes to items nobody saw, and the leak did not manufacture it.

**The ranking inverted.** On the compromised set Fable held severe errors at 10.9% against Sonnet's 14.4%, and that was credited to its rank-2 subspace eraser. Clean: **Sonnet 9.1%, Fable 23.5%** — a 14.4-point reversal. The compromised set's 15.3% base floor left no headroom to separate them; this set starts at 28.9% and separates them immediately.

**Context no longer loses to the best edit on the mean.** Old set: context 68.7 against Sonnet 70.6. Clean: context 63.4 against Sonnet 60.9. The two levers win on different metrics, and the mean alone hides it.

**Kimi is the only arm that moved the model backwards** (severe 28.9 → 30.4). It had concluded the failures were missing knowledge and submitted the published loop fix unchanged.

## The leak, fully characterized

Three independent routes carried held-out content, across **406 paths**:

1. `practice/` — the forward-pass server globbed the whole directory, and the held-out sets lived there. This is the one found first.
2. `exports/` and `gpu/box_sync/work/` — copies made for archival.
3. `work/eval_*/rows_heldout_*.jsonl` — **written by the harness into the editor's own working directory**, carrying both the items and the grader patterns. Closing route 1 alone would not have made a later run clean.

The save-timing audit is what made the arms interpretable: for each editor, when it first read a held-out file against when each `save_edit` landed. Graded-answer counts alone would have cleared Astra, whose submitted weights turn out to be byte-identical to a snapshot taken before its exposure, and would not have flagged that every one of Fable's and Opus's weight-writes came after theirs.

`gpu/fable_08b_clean.sh` now sweeps every matching path out of the editor's tree before the editor starts and aborts if any remains. The sweep matches by pattern, not by name, and restricts the `*heldout*` file match to `*.jsonl` so that `eval_heldout.py` itself is not moved.

## Not measured on this set

- **All seven edit + context combinations.** `remeasure_v2.sh` evaluated nine single arms, the base and context alone; no combo was in its loop. `gpu/remeasure_combos.sh` fills this — pure evaluation, no API spend, 2–3 h.
- **The LoRA.** Excluded because its weights sat in `finetune/merged/` (a full merged copy) rather than as per-tensor `.npy` files under `runs/full_finetune/edited_final/`. `finetune/merged_to_edits.py` converts one to the other — 96 changed projections — and `gpu/start_both.command` now runs it on the Mac before pushing, so only the changed tensors cross the wire. Its old-set result stands at 68.3 / 85.7 / 16.1.
- **The per-family breakdown and the teacher-forced steering index.** The re-measurement reported mean, fabrication and severe-error rate only. The TF prompt set was written against the old item ids and shares none with the `h2_*` items, so it raises `FileNotFoundError`; every clean run passes `--no-tf` until that prompt set is regenerated.
- **A clean Fable *construction*.** Every edit here was re-*measured* clean; four were *built* with access. That is a different and better question, and it is what the sealed re-run asks.

## Provenance

Runner `gpu/remeasure_v2.sh`, log `remeasure_v2_*.log` on the pod. Combos runner `gpu/remeasure_combos.sh`, launcher `gpu/start_combos.command`. Status at any time: `gpu/status.command`, which cross-checks each phase marker against its editor turn count and flags a `FINISHED` marker with fewer than 20 turns as a false finish — `run_compare.sh` skips an arm it cannot run with `run_editor || continue`, so the loop still exits zero.
