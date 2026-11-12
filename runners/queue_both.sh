#!/bin/bash
# Runs ON THE BOX in ~/qwen-0.8b/wsp. Two jobs, in this order, because they cannot share a GPU:
#
#   1. the edit+context combos on the clean set   ~2-3 h   no API spend
#   2. the clean Fable editor arm                 ~17-20 h capped API spend
#
# Combos go first on purpose. They are cheap, they fill the gap in numbers that are already
# being written up, and they stage the replacement set into /opt/wsp_v2 - which survives the
# seal that job 2 performs on eval/. Running them the other way round would work, but a
# failure in the expensive job would delay the cheap answer by most of a day.
cd ~/qwen-0.8b/wsp || { echo "FATAL no ~/qwen-0.8b/wsp"; exit 1; }
LOG=~/qwen-0.8b/queue_both_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
ST=~/qwen-0.8b/BOTH_STATUS
echo "=== queue started $(date -u) ==="; echo "log: $LOG"
step () { echo; echo "======== $* ========"; date -u +%H:%M:%S; }
bail () { echo "ABORT: $*"; echo ABORTED > "$ST"; exit 1; }

if [ ! -x .venv/bin/python ] && [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then
  ln -sfn ~/qwen-4b/wsp/.venv .venv && echo "linked .venv -> ~/qwen-4b/wsp/.venv"
fi
PY=.venv/bin/python
[ -x "$PY" ] || bail "no usable venv"
ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 || bail "no 0.8B weights in ~/qwen-0.8b"

# ---------------------------------------------------------------- PHASE 0
# The clean base row lives only on this pod's ephemeral disk. If the box restarted it is gone,
# and every delta in the combos table is measured against it. Rebuilding it is one eval run
# with no editor and no API spend, so do that rather than failing the whole queue.
echo "P0_BASE_CHECK" > "$ST"
step "PHASE 0 - is the clean base row still here?"
if [ -f work/eval_v2_base/summary.json ]; then
  echo "work/eval_v2_base present - nothing to rebuild"
else
  echo "work/eval_v2_base is GONE (ephemeral disk). Rebuilding it: one eval, ~11 min, no API spend."
  V2=/opt/wsp_v2/eval
  if ! { ls "$V2/heldout"/*.jsonl >/dev/null 2>&1 && grep -ql h2_ "$V2/heldout"/*.jsonl 2>/dev/null; }; then
    SRC=""
    for cand in eval/heldout2 /opt/wsp_sealed_08b/eval/heldout2 /opt/wsp_sealed_08b/eval/heldout; do
      if ls "$cand"/*.jsonl >/dev/null 2>&1 && grep -ql h2_ "$cand"/*.jsonl 2>/dev/null; then SRC="$cand"; break; fi
    done
    [ -n "$SRC" ] || bail "the replacement set is nowhere on this box, so the base row cannot be rebuilt"
    mkdir -p "$V2/heldout" && cp "$SRC"/*.jsonl "$V2/heldout/" || bail "could not stage the replacement set"
    echo "staged the replacement set from $SRC"
  fi
  WSP_EVAL_DIR="$V2" $PY eval_heldout.py --arm v2_base --weights original --n 2 --no-tf \
    || bail "could not rebuild the clean base row"
  echo "base row rebuilt"
fi

# ---------------------------------------------------------------- PHASE 1
echo "P1_COMBOS" > "$ST"
step "PHASE 1 - edit + context combos on the clean set (no API spend)"
[ -f gpu/remeasure_combos.sh ] || bail "gpu/remeasure_combos.sh is not on this box"
bash gpu/remeasure_combos.sh || echo "(combos returned non-zero; its own log has the reason. Continuing to phase 2.)"

# ---------------------------------------------------------------- PHASE 2
echo "P2_FABLE_CLEAN" > "$ST"
step "PHASE 2 - the clean Fable editor arm (held-out sets sealed out of its tree)"
[ -f gpu/fable_08b_clean.sh ] || bail "gpu/fable_08b_clean.sh is not on this box"
bash gpu/fable_08b_clean.sh
rc=$?

step "queue finished, rc=$rc at $(date -u)"
if [ $rc -eq 0 ]; then echo "FINISHED" > "$ST"; else echo "EXIT_$rc" > "$ST"; fi
echo
echo "PULL THE RESULTS. This disk is ephemeral and everything above dies with the pod."
echo "  double-click  ~/qwen-0.8b/wsp/gpu/pull_clean.command"
