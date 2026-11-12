#!/bin/bash
# Runs ON THE BOX in ~/qwen-0.8b/wsp. Re-measures every frozen edit on the 141 replacement items.
# No editor, no API spend: this is pure evaluation.
#
# Unlike the earlier draft of this script, it does NOT overwrite eval/heldout/ with heldout2.
# The original set is what the contaminated numbers were measured on and must survive for
# comparison. Instead heldout2 is staged into a separate store and WSP_EVAL_DIR points at it.
cd ~/qwen-0.8b/wsp || { echo "FATAL no ~/qwen-0.8b/wsp"; exit 1; }
LOG=~/qwen-0.8b/remeasure_v2_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
ST=~/qwen-0.8b/REMEASURE_STATUS
echo "=== heldout2 re-measurement, started $(date -u) ==="; echo "log: $LOG"
fail () { echo "ABORT: $*"; echo "ABORTED" > "$ST"; exit 1; }
step () { echo; echo "===== $* ====="; date -u +%H:%M:%S; }

if [ ! -x .venv/bin/python ] && [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then
  ln -sfn ~/qwen-4b/wsp/.venv .venv && echo "linked .venv -> ~/qwen-4b/wsp/.venv"
fi
PY=.venv/bin/python
[ -x "$PY" ] || fail "no usable venv"
ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 || fail "no 0.8B safetensors in ~/qwen-0.8b"
grep -q 'WSP_EVAL_DIR' eval_heldout.py || fail "eval_heldout.py is not the patched copy"
[ -d eval/heldout2 ] || fail "eval/heldout2 is not on this box"

# --- stage heldout2 under the name the harness expects, WITHOUT touching eval/heldout/ ---
V2=/opt/wsp_v2/eval
mkdir -p "$V2/heldout"
cp eval/heldout2/*.jsonl "$V2/heldout/" || fail "could not stage heldout2"
export WSP_EVAL_DIR="$V2"
# Pair every arm against the v2 base run, not work/eval_base (which holds the OLD set's rows).
export WSP_BASE_ARM=v2_base
# The teacher-forced steering index is SKIPPED on this set, deliberately. Its prompt set
# (eval/steer_tf_heldout.json + _meta.json) was generated for the old steering items and shares
# zero ids with the h2_steer_* items, so reusing it would produce a number that looks fine and
# means nothing. Regenerating it for the replacement set is separate, still-owed work.
TF=--no-tf
echo "staged $(ls "$V2/heldout"/*.jsonl | wc -l | tr -d ' ') files into $V2/heldout (eval/heldout/ untouched)"
$PY -c "
import os,sys; sys.path.insert(0,'.')
import eval_heldout as E
print('  harness will read from:', E.EVAL)
r=E.load_set('heldout_fabrication')
print('  fabrication items:', len(r), '| first id:', r[0]['id'])
assert r[0]['id'].startswith('h2_'), 'NOT the replacement set - ids should start h2_'
print('  confirmed: the replacement set is in use')" || fail "the harness is not reading the replacement set"
echo "RUNNING" > "$ST"

# ---------- A. grader validation (the gate: numbers mean nothing until this is inspected) ----------
step "A. grader validation - base model on the new set, n=1"
$PY eval_heldout.py --arm v2_validate --weights original --n 1 $TF || echo "(validation run failed)"
echo
echo "--- GRADER DIGEST: failures to inspect before trusting any number below ---"
$PY - <<'DIGEST'
import json, glob, os, collections
rows = []
for f in glob.glob("work/eval_v2_validate/rows_*.jsonl"):
    fam = os.path.basename(f)[5:-6]
    for line in open(f):
        if line.strip():
            d = json.loads(line); d["_fam"] = fam; rows.append(d)
if not rows:
    print("  no rows written - the validation run did not produce output"); raise SystemExit
by = collections.defaultdict(lambda: [0, 0])
for d in rows:
    b = by[d["_fam"]]; b[1] += 1; b[0] += int(d.get("ok") or d.get("correct") or 0)
print("  per-family accuracy of the UNEDITED model on the new set (n=1):")
for fam, (ok, n) in sorted(by.items()):
    print(f"    {fam:28s} {ok:3d}/{n:3d} = {100*ok/max(n,1):5.1f}%")
print()
print("  three failures per family - is the GRADER wrong, or the MODEL?")
seen = collections.Counter()
for d in rows:
    if (d.get("ok") or d.get("correct")): continue
    fam = d["_fam"]
    if seen[fam] >= 3: continue
    seen[fam] += 1
    txt = (d.get("text") or d.get("answer") or "").replace("\n", " ")[:220]
    ans = d.get("reference") or d.get("answer") or {}
    print(f"    [{fam}] id={d.get('id')}")
    print(f"       model: {txt}")
    if isinstance(ans, dict):
        if ans.get("must"): print(f"       must: {ans['must'][:3]}")
        if ans.get("must_not"): print(f"       must_not: {ans['must_not'][:3]}")
    if d.get("violated"): print(f"       violated: {d['violated'][:3]}")
DIGEST

# ---------- B. the base row ----------
step "B. base row on the new set, n=2"
$PY eval_heldout.py --arm v2_base --weights original --n 2 $TF || echo "(base failed)"

# ---------- C. every frozen edit ----------
mkdir -p edited
for a in astra_published fable opus astra astra80 astra_xhigh kimi sonnet finetune; do
  SRC="runs/full_$a/edited_final"
  if [ ! -d "$SRC" ]; then
    # Sonnet may have been interrupted at 39/40: its snapshot is still a real artifact
    if [ "$a" = sonnet ] && [ -d runs/full_sonnet/edited_partial ]; then
      SRC=runs/full_sonnet/edited_partial; echo "note: using sonnet's edited_partial (the run was interrupted)"
    else
      echo "skip $a (no edit on this box)"; continue
    fi
  fi
  rm -f edited/*.npy 2>/dev/null
  cp "$SRC"/*.npy edited/ 2>/dev/null
  step "arm $a ($(ls edited/*.npy 2>/dev/null | wc -l | tr -d ' ') tensors from $SRC)"
  $PY eval_heldout.py --arm "v2_$a" --weights edited --n 2 $TF || echo "($a failed)"
done
rm -f edited/*.npy 2>/dev/null

step "D. context arm (original weights + the advising prompt)"
$PY eval_heldout.py --arm v2_context --weights original --n 2 $TF --context contexts/advising.md || echo "(context failed)"

# ---------- the comparison ----------
step "RESULT: replacement set vs the original (contaminated) numbers"
$PY - <<'TABLE'
import json, glob, os
ORIG = {  # the numbers measured on the leaked set
 "base":{"mean":62.5,"fab":57.1,"sev":15.3},  "astra_published":{"mean":59.4,"fab":42.9,"sev":29.2},
 "fable":{"mean":67.9,"fab":89.3,"sev":10.9}, "opus":{"mean":65.7,"fab":85.7,"sev":16.2},
 "astra":{"mean":62.5,"fab":60.7,"sev":24.5}, "astra80":{"mean":62.5,"fab":60.7,"sev":24.5},
 "astra_xhigh":{"mean":65.4,"fab":78.6,"sev":18.0}, "kimi":{"mean":59.4,"fab":42.9,"sev":29.2},
 "context":{"mean":68.7,"fab":78.6,"sev":17.9}, "finetune":{"mean":68.3,"fab":85.7,"sev":16.1},
}
print(f"  {'arm':18s} {'mean':>18s} {'fabrication':>20s} {'severe err':>18s}")
print(f"  {'':18s} {'new   orig   d':>18s} {'new   orig    d':>20s} {'new   orig   d':>18s}")
for d in sorted(glob.glob("work/eval_v2_*/summary.json")):
    arm = os.path.basename(os.path.dirname(d))[8:]
    if arm == "validate": continue
    s = json.load(open(d)); st = s.get("sets", {})
    m = s.get("mean_acc", 0)*100
    fb = st.get("heldout_fabrication", {}).get("acc", 0)*100
    sv = s.get("severe_rate", 0)*100
    o = ORIG.get(arm)
    if o:
        print(f"  {arm:18s} {m:5.1f} {o['mean']:6.1f} {m-o['mean']:+6.1f}"
              f" {fb:7.1f} {o['fab']:6.1f} {fb-o['fab']:+7.1f}"
              f" {sv:6.1f} {o['sev']:6.1f} {sv-o['sev']:+6.1f}")
    else:
        print(f"  {arm:18s} {m:5.1f} {'-':>6s} {'-':>6s} {fb:7.1f} {'-':>6s} {'-':>7s} {sv:6.1f}")
print()
print("  A large negative delta on fabrication = that arm's gain was substantially leak-inflated.")
print("  A small delta = the edit generalises and the leak cost credibility, not substance.")
TABLE

echo; echo "FINISHED" > "$ST"
echo "=== done $(date -u) ==="
echo "Pull from the Mac: double-click ~/qwen-0.8b/wsp/gpu/pull_remeasure.command"
