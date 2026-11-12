#!/bin/bash
# Runs ON THE BOX in ~/qwen-0.8b/wsp. Measures the EDIT + CONTEXT combos, and the LoRA,
# on the 141 provenance-clean replacement items. No editor, no API spend: pure evaluation.
#
# Why this script exists: gpu/remeasure_v2.sh evaluated nine single edits, the base, and
# context alone. Its loop never touched a *_context combo, so the only numbers that exist
# for sonnet_context, fable_context, opus_context and the rest are from the compromised
# set, where the base itself scored 11.8 points higher. Those are not comparable with the
# clean rows and must not be put in the same table.
#
# Arm names are v2_<arm>_context, so gpu/remeasure_v2.sh's own result table picks them up.
cd ~/qwen-0.8b/wsp || { echo "FATAL no ~/qwen-0.8b/wsp"; exit 1; }
LOG=~/qwen-0.8b/remeasure_combos_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
ST=~/qwen-0.8b/COMBOS_STATUS
echo "=== clean edit+context combos, started $(date -u) ==="; echo "log: $LOG"
fail () { echo "ABORT: $*"; echo "ABORTED" > "$ST"; exit 1; }
step () { echo; echo "===== $* ====="; date -u +%H:%M:%S; }

if [ ! -x .venv/bin/python ] && [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then
  ln -sfn ~/qwen-4b/wsp/.venv .venv && echo "linked .venv -> ~/qwen-4b/wsp/.venv"
fi
PY=.venv/bin/python
[ -x "$PY" ] || fail "no usable venv"
ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 || fail "no 0.8B safetensors in ~/qwen-0.8b"
grep -q 'WSP_EVAL_DIR' eval_heldout.py || fail "eval_heldout.py is not the patched copy"
grep -q 'WSP_BASE_ARM' eval_heldout.py || fail "eval_heldout.py lacks the WSP_BASE_ARM patch"
[ -f contexts/advising.md ] || fail "contexts/advising.md is not on this box"

# --- find the replacement set. Phase C's seal MOVES eval/ out of the tree, so look in all
# --- three places it can legitimately be, newest-staged first.
V2=/opt/wsp_v2/eval
if ls "$V2/heldout"/*.jsonl >/dev/null 2>&1 && grep -ql h2_ "$V2/heldout"/*.jsonl 2>/dev/null; then
  echo "using the replacement set already staged at $V2/heldout"
else
  SRC=""
  for cand in eval/heldout2 /opt/wsp_sealed_08b/eval/heldout2 /opt/wsp_sealed_08b/eval/heldout; do
    if ls "$cand"/*.jsonl >/dev/null 2>&1 && grep -ql h2_ "$cand"/*.jsonl 2>/dev/null; then
      SRC="$cand"; break; fi
  done
  [ -n "$SRC" ] || fail "cannot find the replacement set (eval/heldout2, or the sealed copy). Nothing measured."
  mkdir -p "$V2/heldout"
  cp "$SRC"/*.jsonl "$V2/heldout/" || fail "could not stage the replacement set from $SRC"
  echo "staged $(ls "$V2/heldout"/*.jsonl | wc -l | tr -d ' ') files from $SRC into $V2/heldout"
fi
export WSP_EVAL_DIR="$V2"
export WSP_BASE_ARM=v2_base        # deltas are against the clean base, not the old one

# --- prove the harness is reading the replacement set before anything is measured
$PY -c "
import os,sys; sys.path.insert(0,'.')
import eval_heldout as E
print('  harness reads from:', E.EVAL)
r=E.load_set('heldout_fabrication')
print('  fabrication items:', len(r), '| first id:', r[0]['id'])
assert r[0]['id'].startswith('h2_'), 'NOT the replacement set - ids should start h2_'
print('  confirmed: the replacement set is in use')" || fail "the harness is not reading the replacement set"

[ -d work/eval_v2_base ] || fail "work/eval_v2_base is missing: without the clean base row there is nothing to take deltas against"
echo "clean base row present: work/eval_v2_base"
echo "RUNNING" > "$ST"

# --- --no-tf on every call, deliberately. The teacher-forced steering prompts were written
# --- for the original set and share ZERO ids with the h2_ items, so asking for the TF index
# --- raises FileNotFoundError and loses the whole arm. index_pp is a secondary diagnostic;
# --- mean, fabrication and severe-error rate are unaffected.
TF=--no-tf

mkdir -p edited
RAN=0; SKIPPED=""
for a in sonnet fable opus astra astra80 astra_xhigh kimi; do
  SRC="runs/full_$a/edited_final"
  if [ ! -d "$SRC" ] && [ "$a" = sonnet ] && [ -d runs/full_sonnet/edited_partial ]; then
    SRC=runs/full_sonnet/edited_partial; echo "note: using sonnet's edited_partial (that run was interrupted at 39/40)"
  fi
  [ -d "$SRC" ] || { echo "skip ${a}_context (no edit on this box)"; SKIPPED="$SKIPPED ${a}_context"; continue; }
  [ -f "work/eval_v2_${a}_context/summary.json" ] && { echo "skip ${a}_context (already measured on the clean set)"; continue; }
  rm -f edited/*.npy 2>/dev/null
  cp "$SRC"/*.npy edited/ 2>/dev/null
  n=$(ls edited/*.npy 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || { echo "skip ${a}_context (no tensors copied from $SRC)"; SKIPPED="$SKIPPED ${a}_context"; continue; }
  step "arm v2_${a}_context  ($n tensors from $SRC, plus contexts/advising.md)"
  if $PY eval_heldout.py --arm "v2_${a}_context" --weights edited --n 2 \
        --context contexts/advising.md $TF; then RAN=$((RAN+1))
  else echo "(${a}_context FAILED - see above)"; SKIPPED="$SKIPPED ${a}_context"; fi
done

# --- the LoRA arm, which remeasure_v2.sh skipped for want of runs/full_finetune/edited_final
step "LoRA arm on the clean set"
if [ -f work/eval_v2_finetune/summary.json ]; then
  echo "already measured on the clean set"
elif [ -d runs/full_finetune/edited_final ]; then
  rm -f edited/*.npy 2>/dev/null; cp runs/full_finetune/edited_final/*.npy edited/ 2>/dev/null
  $PY eval_heldout.py --arm v2_finetune --weights edited --n 2 $TF && RAN=$((RAN+1)) || SKIPPED="$SKIPPED finetune"
else
  echo "no runs/full_finetune/edited_final on this box - the LoRA is stored as adapter weights,"
  echo "not as a dense .npy diff, so it needs its own loader. Left unmeasured."
  SKIPPED="$SKIPPED finetune"
fi
rm -f edited/*.npy 2>/dev/null

# ---------------------------------------------------------------- the result
step "RESULT: clean set, single levers beside their edit+context combos"
$PY - <<'TABLE'
import json, glob, os
def row(arm):
    p = f"work/eval_v2_{arm}/summary.json"
    if not os.path.exists(p): return None
    s = json.load(open(p)); st = s.get("sets", {})
    return (s.get("mean_acc",0)*100,
            st.get("heldout_fabrication",{}).get("acc",0)*100,
            s.get("severe_rate",0)*100)
b = row("base")
if not b:
    print("  no clean base row - cannot take deltas"); raise SystemExit
ctx = row("context")
print(f"  base (no lever)        {b[0]:6.1f} {b[1]:7.1f} {b[2]:8.1f}")
if ctx:
    print(f"  context only           {ctx[0]:6.1f} {ctx[1]:7.1f} {ctx[2]:8.1f}"
          f"   ({ctx[0]-b[0]:+.1f} / {ctx[1]-b[1]:+.1f} / {ctx[2]-b[2]:+.1f})")
print()
print(f"  {'arm':22s} {'mean':>6s} {'fab':>7s} {'severe':>8s}   {'edit alone -> edit+context':>34s}")
for a in ["sonnet","fable","opus","astra","astra80","astra_xhigh","kimi"]:
    e, c = row(a), row(f"{a}_context")
    if not e and not c: continue
    if e: print(f"  {a:22s} {e[0]:6.1f} {e[1]:7.1f} {e[2]:8.1f}")
    if c:
        d = f"{c[0]-e[0]:+6.1f} {c[1]-e[1]:+7.1f} {c[2]-e[2]:+8.1f}" if e else " " * 22
        print(f"  {a+'_context':22s} {c[0]:6.1f} {c[1]:7.1f} {c[2]:8.1f}   {d}")
    else:
        print(f"  {a+'_context':22s} {'-':>6s} {'-':>7s} {'-':>8s}   not measured")
f = row("finetune")
if f:
    print()
    print(f"  {'finetune (LoRA)':22s} {f[0]:6.1f} {f[1]:7.1f} {f[2]:8.1f}"
          f"   ({f[0]-b[0]:+.1f} / {f[1]-b[1]:+.1f} / {f[2]-b[2]:+.1f})")
print()
print("  Every figure here is on the 141 replacement items, against the clean base.")
print("  Do NOT place these beside the old-set numbers: that set's base scored 11.8 higher.")
TABLE

echo
echo "arms measured this run: $RAN"
[ -z "$SKIPPED" ] && echo "nothing skipped" || echo "SKIPPED or failed:$SKIPPED"
echo "FINISHED" > "$ST"
echo "=== done $(date -u) ==="
