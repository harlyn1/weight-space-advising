#!/bin/bash
# Runs ON THE BOX, inside ~/qwen-0.8b/wsp. Validates the replacement graders against base-model output,
# then re-measures every frozen arm on the new set. Uses the 4B tree's venv; the 0.8B harness needs no
# patch because its hardcoded shapes are the 0.8B's own.
cd ~/qwen-0.8b/wsp || exit 1
PY=~/qwen-4b/wsp/.venv/bin/python
step () { echo; echo "===== $* ====="; date -u +%H:%M:%S; }

cp eval/heldout2/*.jsonl eval/heldout/      # same set names, new items: every eval below uses them
step "A. grader validation: base model on the new set, n=1"
$PY eval_heldout.py --arm v2_validate --weights original --n 1 --no-tf || echo "(validation failed)"

step "B. base row on the new set, n=2"
$PY eval_heldout.py --arm v2_base --weights original --n 2 || echo "(base failed)"

for a in astra_published fable opus astra astra80 astra_xhigh kimi sonnet; do
  [ -d "runs/full_$a/edited_final" ] || { echo "skip $a (no edited_final)"; continue; }
  rm -rf edited; mkdir -p edited; cp runs/full_$a/edited_final/*.npy edited/ 2>/dev/null
  step "arm $a ($(ls edited | wc -l | tr -d ' ') tensors)"
  $PY eval_heldout.py --arm "v2_$a" --weights edited --n 2 || echo "($a failed)"
done
rm -rf edited; mkdir -p edited

step "C. context arm"
$PY eval_heldout.py --arm v2_context --weights original --n 2 --context contexts/advising.md || echo "(context failed)"

step "doneV2"
echo "summaries: work/eval_v2_*/summary.json"
