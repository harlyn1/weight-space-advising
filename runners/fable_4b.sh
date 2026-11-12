#!/bin/bash
# Runs ON THE BOX, inside ~/qwen-4b/wsp. Pod-resident: once started it does not need the Mac.
# Launched by gpu/start_fable_4b.command. Arms: base (paired floor) -> fable (40 passes) -> fable_context (free).
cd ~/qwen-4b/wsp || { echo "FATAL no ~/qwen-4b/wsp"; exit 1; }
LOG=~/qwen-4b/fable_4b_$(date -u +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== Fable 4B arm, started $(date -u) ==="; echo "log: $LOG"

fail () { echo "ABORT: $*"; echo "ABORTED" > ~/qwen-4b/FABLE_4B_STATUS; exit 1; }

# --- preflight: refuse to run if the held-out sets are reachable by the editor ---
grep -q "held-out ids are not available to the editor" server.py || fail "leak guard missing from server.py on the box. Not launching: this is the fault that invalidated the first experiment."
grep -q "def load_eval_items" server.py || fail "load_eval_items missing from server.py (the evaluation would not find the sealed sets)."
if ls practice/ 2>/dev/null | grep -q '^heldout'; then fail "practice/ contains heldout files - the editor could address them by id."; fi
[ -d eval/heldout ] || fail "eval/heldout/ missing on the box; nothing to evaluate against."
[ -f tasks/q2_full_4b.md ] || fail "tasks/q2_full_4b.md missing (the 4B brief was not copied)."
grep -q "Qwen3.5-4B" tasks/q2_full_4b.md || fail "the brief on the box is not the 4B brief."
[ -f envfile.py ] || fail "envfile.py missing; editor_api.py imports it and would crash on turn 1."
[ -x .venv/bin/python ] || fail "no venv at ~/qwen-4b/wsp/.venv"
# key presence only, never the value
if [ -f .env ]; then
  n=$(grep -c '^ANTHROPIC_API_KEY=.\{20,\}' .env || true)
  [ "$n" -ge 1 ] || fail "ANTHROPIC_API_KEY not present (or too short) in .env on the box."
  echo "preflight: ANTHROPIC_API_KEY present in .env (value not read)"
else fail ".env missing on the box."; fi
.venv/bin/python -c "import kernel,sys; s=getattr(kernel,'_arch_string',None); print('arch:', s() if s else 'NO _arch_string - harness not patched for 4B')" || fail "kernel.py will not import"
# --- SEAL: move the held-out sets out of the editor's tree entirely. The id guard only covers
# forward requests; at 0.8B the actual leak was a python cell opening the files directly. ---
SEALED=/opt/wsp_sealed
mkdir -p "$SEALED"
# the WHOLE eval/ tree moves: heldout2 (the replacement set) is as sensitive as heldout
if [ -d eval ]; then
  rm -rf "$SEALED/eval"; mv eval "$SEALED/eval"
  echo "sealed: eval/ -> $SEALED/eval ($(find "$SEALED/eval" -name '*.jsonl' | wc -l | tr -d ' ') jsonl files)"
fi
export WSP_EVAL_DIR="$SEALED/eval"
[ -d "$WSP_EVAL_DIR/heldout" ] || fail "sealed store has no heldout/ - the evaluation would find nothing"
leftover=$(find ~/qwen-4b/wsp -iname '*heldout*' -o -iname 'ho_*' 2>/dev/null | head -5)
[ -z "$leftover" ] || fail "held-out files still reachable inside the editor's tree:\n$leftover"
grep -rl 'ho_[a-z]' ~/qwen-4b/wsp/practice/ 2>/dev/null | head -1 | grep -q . && fail "practice/ still contains ho_ ids"
echo "preflight: no held-out file is reachable under ~/qwen-4b/wsp"
echo "preflight: OK"; echo "RUNNING" > ~/qwen-4b/FABLE_4B_STATUS

# --- the run. Spend cap is enforced inside editor_api.py (WSP_MAX_COST_USD). ---
export ARMS="base fable fable_context"
export BUDGET=${BUDGET:-40}
export EFFORT=${EFFORT:-max}
export TASK=tasks/q2_full_4b.md
export WSP_MAX_COST_USD=${WSP_MAX_COST_USD:-200}
export EVAL_N=${EVAL_N:-2}
export SERVER_BATCH=${SERVER_BATCH:-32}
export EVAL_BATCH=${EVAL_BATCH:-32}
echo "arms=$ARMS budget=$BUDGET effort=$EFFORT cap=\$$WSP_MAX_COST_USD brief=$TASK"
bash run_compare.sh
rc=$?
echo "=== run_compare.sh exited rc=$rc at $(date -u) ==="
echo "--- spend ---"; grep -h -o 'cumulative \$[0-9.]*' runs/full_fable.log 2>/dev/null | tail -3
echo "--- summaries present ---"; ls -d work/eval_* 2>/dev/null
echo "--- LEAK AUDIT (built in, so we know rather than find out later) ---"
T=runs/full_fable/tool_calls.jsonl
if [ -f "$T" ]; then
  .venv/bin/python - "$T" <<'AUDIT'
import json,re,sys
HO=re.compile(r'\bho_[a-z0-9_]+'); SEAL=re.compile(r'heldout|wsp_sealed|WSP_EVAL_DIR')
hits=[]
for line in open(sys.argv[1]):
    try: d=json.loads(line)
    except: continue
    c=d.get("code","") or ""
    if HO.search(c) or SEAL.search(c): hits.append((d.get("turn"), (HO.search(c) or SEAL.search(c)).group(0)))
print(f"cells referencing a held-out id or the sealed store: {len(hits)}")
for t,w in hits[:12]: print(f"   turn {t}: {w}")
print("CLEAN" if not hits else "REVIEW THESE TURNS BEFORE REPORTING ANY NUMBER")
AUDIT
fi
echo "--- restoring the sealed sets to the tree ---"
[ -d /opt/wsp_sealed/eval ] && { rm -rf eval; cp -r /opt/wsp_sealed/eval eval; echo "restored eval/ into the tree"; }
if [ $rc -eq 0 ]; then echo "FINISHED" > ~/qwen-4b/FABLE_4B_STATUS; else echo "EXIT_$rc" > ~/qwen-4b/FABLE_4B_STATUS; fi
echo "done. pull with: double-click ~/qwen-0.8b/wsp/gpu/pull_4b.command on the Mac"
