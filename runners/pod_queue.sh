#!/bin/bash
# Runs ON THE BOX in ~/qwen-0.8b/wsp. Three phases, in this order for a reason:
#
#   A. the three missing context arms   - needs eval/heldout/ PRESENT   ~1-2 h, no API spend
#   B. the heldout2 re-measurement      - uses a separately staged set  ~2 h,   no API spend
#   C. the clean Fable editor arm       - needs eval/heldout/ SEALED    ~17-20 h, capped API spend
#
# They cannot run in parallel on one box: every phase writes to the shared edited/ directory,
# and run_compare.sh's stop_server kills by pattern across folders. Sequencing is not a
# limitation to work around, it is what keeps them from corrupting each other.
#
# The cheap, informative work finishes first, so a morning check shows A and B complete with
# C in progress - and C is spend-capped, so it can be stopped on what A and B reveal.
cd ~/qwen-0.8b/wsp || { echo "FATAL no ~/qwen-0.8b/wsp"; exit 1; }
LOG=~/qwen-0.8b/pod_queue_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
ST=~/qwen-0.8b/POD_QUEUE_STATUS
echo "=== pod queue started $(date -u) ==="; echo "log: $LOG"
step () { echo; echo "======== $* ========"; date -u +%H:%M:%S; }
bail () { echo "ABORT: $*"; echo ABORTED > "$ST"; exit 1; }

if [ ! -x .venv/bin/python ] && [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then
  ln -sfn ~/qwen-4b/wsp/.venv .venv && echo "linked .venv -> ~/qwen-4b/wsp/.venv"
fi
PY=.venv/bin/python
[ -x "$PY" ] || bail "no venv"
ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 || bail "no 0.8B weights in ~/qwen-0.8b"
[ -d eval/heldout ] || bail "eval/heldout/ missing - phase A needs it present"
[ -f contexts/advising.md ] || bail "contexts/advising.md missing"

# ------------------------------------------------------------------ PHASE A
# Phase A is OFF by default as of 2026-11-11: the Mac finished astra80_context,
# astra_xhigh_context and sonnet_context overnight, so running them here would redo
# 1-2 h of completed work. Set RUN_PHASE_A=1 to force it.
if [ "${RUN_PHASE_A:-0}" != 1 ]; then
  echo "PHASE A skipped - those three context arms were completed on the Mac."
  echo "   (RUN_PHASE_A=1 to force them)"
else
echo "A_CONTEXT_ARMS" > "$ST"
step "PHASE A - the three missing context arms, on the ORIGINAL held-out set"
echo "Each took ~5 h on the Mac: the 1.3k-token context pushes the linear-attention kernels onto"
echo "the reference PyTorch path. On CUDA it is minutes, so the teacher-forced steering index is"
echo "computed here (--tf-with-context) instead of being skipped as it was for the earlier combos."
mkdir -p edited
for a in astra80 astra_xhigh sonnet; do
  SRC="runs/full_$a/edited_final"
  if [ ! -d "$SRC" ] && [ "$a" = sonnet ] && [ -d runs/full_sonnet/edited_partial ]; then
    SRC=runs/full_sonnet/edited_partial; echo "note: using sonnet's edited_partial"
  fi
  [ -d "$SRC" ] || { echo "skip ${a}_context (no edit for $a on this box)"; continue; }
  [ -f "work/eval_${a}_context/summary.json" ] && { echo "skip ${a}_context (already evaluated)"; continue; }
  rm -f edited/*.npy 2>/dev/null
  cp "$SRC"/*.npy edited/ 2>/dev/null
  step "arm ${a}_context ($(ls edited/*.npy 2>/dev/null | wc -l | tr -d ' ') tensors from $SRC)"
  $PY eval_heldout.py --arm "${a}_context" --weights edited --n 2 \
      --context contexts/advising.md --tf-with-context || echo "(${a}_context failed)"
done
rm -f edited/*.npy 2>/dev/null

step "PHASE A results - edit + context combos"
$PY - <<'TABLE'
import json, glob, os
print(f"  {'arm':24s} {'mean':>7s} {'fab':>7s} {'severe':>8s} {'tf idx':>8s}")
for d in sorted(glob.glob("work/eval_*_context/summary.json")):
    arm = os.path.basename(os.path.dirname(d))[5:]
    s = json.load(open(d)); st = s.get("sets", {})
    tf = (s.get("steering_tf") or {}).get("index_pp")
    print(f"  {arm:24s} {s.get('mean_acc',0)*100:7.1f} "
          f"{st.get('heldout_fabrication',{}).get('acc',0)*100:7.1f} "
          f"{s.get('severe_rate',0)*100:8.1f} {str(tf):>8s}")
print("  reference: fable_context 71.7 / 100.0 / 11.4 / -8.1    opus_context 72.3 / 82.1 / 16.2 / skipped")
print("  base 62.5 / 57.1 / 15.3 / -9.0")
TABLE
fi

# ------------------------------------------------------------------ PHASE B
echo "B_REMEASURE" > "$ST"
step "PHASE B - re-measuring every frozen edit on the 141 replacement items"
if [ -f gpu/remeasure_v2.sh ]; then
  bash gpu/remeasure_v2.sh || echo "(re-measurement returned non-zero; continuing to phase C)"
else
  echo "gpu/remeasure_v2.sh not on this box - phase B skipped"
fi

# ------------------------------------------------------------------ PHASE C
echo "C_FABLE_CLEAN" > "$ST"
step "PHASE C - the clean Fable editor arm (held-out sets sealed out of its tree)"
[ -f gpu/fable_08b_clean.sh ] || bail "gpu/fable_08b_clean.sh is not on this box"
bash gpu/fable_08b_clean.sh
rc=$?

step "pod queue finished, rc=$rc at $(date -u)"
if [ $rc -eq 0 ]; then echo "FINISHED" > "$ST"; else echo "EXIT_$rc" > "$ST"; fi
echo "Pull from the Mac: double-click ~/qwen-0.8b/wsp/gpu/pull_clean.command"
