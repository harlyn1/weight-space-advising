#!/bin/bash
# Runs ON THE BOX in ~/qwen-0.8b/wsp. A provenance-clean re-run of the Fable 0.8B arm:
# same model, same brief, same starting point as the original - but with the held-out sets
# moved out of the editor's tree before it starts. Pod-resident; needs nothing from the Mac.
#
# Deliberately contains no 'rm -rf': the held-out sets are MOVED, never deleted, and the
# restore step refuses to overwrite anything that is already in place.
cd ~/qwen-0.8b/wsp || { echo "FATAL no ~/qwen-0.8b/wsp"; exit 1; }
LOG=~/qwen-0.8b/fable_clean_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== Fable 0.8B CLEAN re-run, started $(date -u) ==="; echo "log: $LOG"
ST=~/qwen-0.8b/FABLE_CLEAN_STATUS
fail () { echo "ABORT: $*"; echo "ABORTED" > "$ST"; exit 1; }

# --- venv: the 0.8B tree's own .venv is excluded from push.sh; reuse the 4B tree's ---
if [ ! -x .venv/bin/python ] && [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then
  ln -sfn ~/qwen-4b/wsp/.venv .venv && echo "linked .venv -> ~/qwen-4b/wsp/.venv"
fi
[ -x .venv/bin/python ] || fail "no usable venv for ~/qwen-0.8b/wsp"

# --- preflight ---
ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 || fail "no 0.8B safetensors in ~/qwen-0.8b (kernel.MODEL_DIR). Run gpu/push.sh from the Mac."
[ "$(ls work/best_paper/*.npy 2>/dev/null | wc -l)" -eq 2 ] || fail "work/best_paper/ must hold the 2 published loop-fix tensors (this arm's starting point)"
[ -f tasks/q2_full.md ] || fail "tasks/q2_full.md missing (the original brief)"
grep -q "held-out ids are not available to the editor" server.py || fail "leak guard missing from server.py"
grep -q 'WSP_EVAL_DIR' server.py || fail "server.py is not the patched copy (no WSP_EVAL_DIR support)"
grep -q 'WSP_EVAL_DIR' eval_heldout.py || fail "eval_heldout.py is not the patched copy"
grep -q 'fable_clean' run_compare.sh || fail "run_compare.sh is not the patched copy (fable_clean would not route to Fable)"
[ -d runs/full_fable_clean ] && fail "runs/full_fable_clean already exists - move it aside first, or this resumes rather than starting clean"
n=$(grep -c '^ANTHROPIC_API_KEY=.\{20,\}' .env 2>/dev/null || true)
[ "${n:-0}" -ge 1 ] || fail "ANTHROPIC_API_KEY not present in .env"
echo "preflight: weights, loop-fix seed, brief, patched harness, key - all present"

# --- SEAL: remove every copy of the held-out data from the editor's tree. ---
# History: naming the locations failed three times. _leak_quarantine, then
# exports/_leak_quarantine, then exports/.../evalsets/heldout. A sweep of the whole tree
# then found 406 held-out paths, including work/eval_*/rows_heldout_*.jsonl inside the
# editor's OWN work folder - those carry the items and the grader's must/must_not.
# So: match by pattern, never by name, and never touch eval_heldout.py (hence *.jsonl only).
SEALED=/opt/wsp_sealed_08b
STAMP=$(date -u +%Y%m%d_%H%M%S)
VAULT="$SEALED/swept_$STAMP"
mkdir -p "$SEALED" "$VAULT"

hold () {   # move $1 into the vault, keeping its relative path so it can be put back
  [ -e "$1" ] || return 0
  d="$VAULT/$(dirname "$1")"
  mkdir -p "$d" && mv "$1" "$d/" && echo "  sealed  $1"
}

# a) the canonical eval tree keeps a stable name: the evaluation reads it via WSP_EVAL_DIR
if [ -d eval ]; then
  if [ -d "$SEALED/eval" ]; then mv eval "$SEALED/eval_intree_$STAMP" && echo "  sealed  eval/ (canonical copy already held)"
  else mv eval "$SEALED/eval" && echo "  sealed  eval/ -> $SEALED/eval"; fi
fi
export WSP_EVAL_DIR="$SEALED/eval"
[ -d "$WSP_EVAL_DIR/heldout" ] || fail "sealed store has no heldout/ - the evaluation would find nothing"

# b) prior results and bundles that are known to carry held-out items
for p in exports gpu/box_sync _leak_quarantine work/eval_* work/gen_ev_* work/fwd_ev_*; do hold "$p"; done

# c) backstop: anything still matching anywhere. Directories named *heldout*, and *.jsonl
#    files - never *.py or *.pyc, so the harness itself is safe.
find . \( -type d -iname '*heldout*' -o -type f -iname '*heldout*.jsonl' \) 2>/dev/null \
  | sed 's|^\./||' | while IFS= read -r p; do hold "$p"; done

# d) verify by the same matcher, and refuse to run if anything survived
leftover=$(find . \( -type d -iname '*heldout*' -o -type f -iname '*heldout*.jsonl' \) 2>/dev/null | head -5)
[ -z "$leftover" ] || fail "held-out data still in the editor's tree:\n$leftover"
grep -rl 'ho_[a-z]' practice/ 2>/dev/null | head -1 | grep -q . && fail "practice/ still contains ho_ ids"
echo "preflight: swept $(find "$VAULT" -type f 2>/dev/null | wc -l | tr -d ' ') held-out files out of the tree; none remain"
echo "RUNNING" > "$ST"

# --- the run ---
export ARMS="fable_clean"
export BUDGET=${BUDGET:-40} EFFORT=${EFFORT:-max} TASK=tasks/q2_full.md
export WSP_MAX_COST_USD=${WSP_MAX_COST_USD:-250}
export EVAL_N=${EVAL_N:-2} SERVER_BATCH=${SERVER_BATCH:-32} EVAL_BATCH=${EVAL_BATCH:-32}
echo "arm=fable_clean budget=$BUDGET effort=$EFFORT cap=\$$WSP_MAX_COST_USD brief=$TASK"
bash run_compare.sh
rc=$?
echo "=== run_compare.sh exited rc=$rc at $(date -u) ==="

# --- built-in leak audit: know now, rather than find out five days later ---
echo "--- LEAK AUDIT ---"
T=runs/full_fable_clean/tool_calls.jsonl
if [ -f "$T" ]; then
  .venv/bin/python - "$T" <<'AUDIT'
import json, re, sys
HO = re.compile(r'\bho_[a-z0-9_]+')
SEAL = re.compile(r'heldout|wsp_sealed|WSP_EVAL_DIR')
SAVE = re.compile(r'\bsave_edit\s*\(')
hits, saves = [], []
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except Exception:
        continue
    c = d.get("code", "") or ""
    t = d.get("turn")
    if SAVE.search(c):
        saves.append(t)
    m = HO.search(c) or SEAL.search(c)
    if m:
        hits.append((t, m.group(0)))
lo = saves[0] if saves else "-"
hi = saves[-1] if saves else "-"
print(f"save_edit calls: {len(saves)} (turns {lo}..{hi})")
print(f"cells referencing a held-out id or the sealed store: {len(hits)}")
for t, w in hits[:12]:
    print(f"   turn {t}: {w}")
print("VERDICT: CLEAN - every weight-write was made with no held-out access" if not hits
      else "VERDICT: REVIEW THESE TURNS BEFORE REPORTING ANY NUMBER")
AUDIT
else
  echo "(no transcript: the arm did not start)"
fi

echo "--- result ---"
if [ -f work/eval_fable_clean/summary.json ]; then
  .venv/bin/python -c "
import json
s=json.load(open('work/eval_fable_clean/summary.json')); st=s.get('sets',{})
print('  CLEAN  mean %.1f  severe %.1f  tf %s'%(s.get('mean_acc',0)*100, s.get('severe_rate',0)*100, (s.get('steering_tf') or {}).get('index_pp')))
print('  '+'  '.join('%s=%.1f'%(k.replace('heldout_','').replace('heldout','facts'), v.get('acc',0)*100) for k,v in st.items()))
print('  ORIG (contaminated): mean 67.9  fabrication 89.3  severe 10.9  tf -2.1')"
fi
grep -h -o 'cumulative \$[0-9.]*' runs/full_fable_clean.log 2>/dev/null | tail -2

echo "--- restoring eval/ into the tree ---"
if [ ! -d eval ] && [ -d "$SEALED/eval" ]; then
  cp -r "$SEALED/eval" eval && echo "restored eval/ (sealed copy kept at $SEALED/eval)"
fi
if [ -d "$VAULT" ]; then
  ( cd "$VAULT" && find . -mindepth 1 -maxdepth 2 -print ) >/dev/null 2>&1
  ( cd "$VAULT" && tar cf - . ) | tar xf - -C . 2>/dev/null \
    && echo "restored $(find "$VAULT" -type f 2>/dev/null | wc -l | tr -d ' ') swept files into the tree"
fi
if [ $rc -eq 0 ]; then echo "FINISHED" > "$ST"; else echo "EXIT_$rc" > "$ST"; fi
echo "done. Pull from the Mac by double-clicking ~/qwen-0.8b/wsp/gpu/pull_clean.command"
