#!/bin/bash
# DOUBLE-CLICK ANY TIME. Prints everything about what is and isn't running on the WSP pod.
# Read-only: starts nothing, stops nothing, changes nothing.
HOST=root@POD_IP
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

printf '\n\033[1m=== WSP pod status — %s ===\033[0m\n' "$(date -u '+%H:%M UTC %d %b')"

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || {
  printf '\n\033[31mPod unreachable.\033[0m Either it is stopped in the RunPod console, or the\n'
  printf 'network is down. Nothing is running if you cannot reach it.\n\n'
  echo "Press return to close."; read _; exit 1; }

$SSH "$HOST" '
echo
echo "--- is anything actually running? ---"
P=$(pgrep -f "[e]ditor_api\.py --task")
S=$(pgrep -f "[s]erver\.py --budget")
if [ -n "$P" ]; then
  for p in $P; do echo "  EDITOR  pid $p  up $(ps -o etime= -p $p | tr -d " ")  cwd $(readlink /proc/$p/cwd)"; done
else echo "  EDITOR  none"; fi
if [ -n "$S" ]; then
  for p in $S; do echo "  SERVER  pid $p  up $(ps -o etime= -p $p | tr -d " ")  cwd $(readlink /proc/$p/cwd)"; done
else echo "  SERVER  none"; fi
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed "s/^/  GPU     /"

echo
echo "--- phase markers ---"
for f in ~/qwen-0.8b/BOTH_STATUS ~/qwen-0.8b/POD_QUEUE_STATUS ~/qwen-0.8b/FABLE_CLEAN_STATUS ~/qwen-0.8b/REMEASURE_STATUS ~/qwen-0.8b/COMBOS_STATUS ~/qwen-4b/FABLE_4B_STATUS; do
  [ -f "$f" ] && printf "  %-22s %s  (written %s)\n" "$(basename $f)" "$(cat $f)" "$(date -u -r "$f" +%H:%M)"
done
echo "  NOTE: EXIT_n / ABORTED / FINISHED are FINISHED states, not running ones."
echo "        BOTH_STATUS runs P0_BASE_CHECK -> P1_COMBOS -> P2_FABLE_CLEAN -> FINISHED."

echo
echo "--- Fable clean arm progress ---"
L=~/qwen-0.8b/wsp/runs/full_fable_clean.log   # phase 2 of the queue writes here too
if [ -f "$L" ]; then
  echo "  turns: $(grep -c "^\[turn" "$L") of ~557 (Fable needed 557 at 0.8B)"
  echo "  spend: $(grep -o "cumulative \$[0-9.]*" "$L" | tail -1)"
  echo "  last:  $(tail -1 "$L" | cut -c1-100)"
  echo "  passes: $(python3 -c "import json;print(json.load(open(\"$HOME/qwen-0.8b/wsp/state.json\"))[\"passes_used\"])" 2>/dev/null || echo "n/a") of 40"
else echo "  not started (no runs/full_fable_clean.log)"; fi

echo
echo "--- re-measurement (phase B) ---"
N=$(ls ~/qwen-0.8b/wsp/work/eval_v2_*/summary.json 2>/dev/null | wc -l)
echo "  arms with a summary: $N  (10 = complete; 0 = every arm crashed)"
RL=$(ls -t ~/qwen-0.8b/remeasure_v2_*.log 2>/dev/null | head -1)
[ -n "$RL" ] && echo "  newest log: $(basename $RL)  ($(date -u -r "$RL" +%H:%M))"

echo
echo "--- edit+context combos on the clean set ---"
N=$(ls -d ~/qwen-0.8b/wsp/work/eval_v2_*_context 2>/dev/null | wc -l)
echo "  combos with a result dir: $N  (7 = all of them)"
for a in sonnet fable opus astra astra80 astra_xhigh kimi; do
  if [ -f ~/qwen-0.8b/wsp/work/eval_v2_${a}_context/summary.json ]; then echo "    ${a}_context  done"; fi
done
CL=$(ls -t ~/qwen-0.8b/remeasure_combos_*.log 2>/dev/null | head -1)
if [ -n "$CL" ]; then
  echo "  newest log: $(basename $CL)  ($(date -u -r "$CL" +%H:%M))"
  A=$(grep -h "^ABORT:" "$CL" 2>/dev/null | tail -1)
  [ -n "$A" ] && echo "  $A"
  echo "  last:  $(tail -1 "$CL" | cut -c1-100)"
else echo "  not started (no remeasure_combos log)"; fi

echo
echo "--- latest abort reason, if any ---"
# the old version piped grep into sed, so "|| echo none" never fired and a no-match
# printed nothing at all. Check explicitly instead.
L=$(ls -t ~/qwen-0.8b/fable_clean_*.log 2>/dev/null | head -1)
if [ -n "$L" ]; then
  A=$(grep -h "^ABORT:" "$L" 2>/dev/null | tail -1)
  if [ -n "$A" ]; then echo "  $(basename $L): $A"
  else echo "  $(basename $L): no ABORT line (it may have run or died another way)"; fi
else echo "  no fable_clean log on the box"; fi

echo
echo "--- marker sanity: does the evidence match the marker? ---"
# run_compare.sh skips an arm it cannot run (missing key, missing dep) with
# "run_editor || continue", so the loop still exits 0 and the runner writes FINISHED.
# A FINISHED marker with no turns is a false finish.
chk () {   # $1 label  $2 status file  $3 editor log
  [ -f "$2" ] || return
  st=$(cat "$2"); t=0
  [ -f "$3" ] && t=$(grep -c "^\[turn" "$3" 2>/dev/null)
  if [ "$st" = FINISHED ] && [ "${t:-0}" -lt 20 ]; then
    echo "  $1: marker says FINISHED but only $t editor turns -> FALSE FINISH, the arm was skipped"
  else
    echo "  $1: $st, $t editor turns"
  fi
}
chk "0.8B fable_clean" ~/qwen-0.8b/FABLE_CLEAN_STATUS ~/qwen-0.8b/wsp/runs/full_fable_clean.log
chk "4B fable"         ~/qwen-4b/FABLE_4B_STATUS      ~/qwen-4b/wsp/runs/full_fable.log
N=$(ls ~/qwen-0.8b/wsp/work/eval_v2_*/summary.json 2>/dev/null | wc -l)
[ -f ~/qwen-0.8b/REMEASURE_STATUS ] && echo "  remeasure: $(cat ~/qwen-0.8b/REMEASURE_STATUS), $N arm summaries written"
' < /dev/null

cat <<'TAIL'

---------------------------------------------------------
Reading this:
  EDITOR none + a FINISHED-state marker  -> nothing running
  EDITOR present + C_FABLE_CLEAN/P2      -> Fable is working
  EVAL present + P1_COMBOS               -> combos are working
  turns climbing between two runs of this -> real progress
---------------------------------------------------------
TAIL
echo "Press return to close."; read _
