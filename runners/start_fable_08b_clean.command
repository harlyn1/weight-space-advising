#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Starts a provenance-clean re-run of the Fable 0.8B arm on the pod.
# Once it reports "launched" the run lives on the POD: close this window, close the Mac, lose wifi.
#
# This version REPAIRS the pod instead of only checking it. The previous one verified six
# conditions in a single && chain and died with one message naming all six, which is useless
# when the pod's disk is ephemeral and the thing actually missing is the 1.6 GB of weights.
#
# THIS IS THE WEIGHT-SPACE-PROGRAMMING POD. It is NOT the insurance RSI pod.
HOST=root@69.8.146.173
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
CAP=${CAP:-250}            # hard spend cap in USD (the original arm cost $110.73)
BUDGET=${BUDGET:-40}       # forward passes, same as the original

SRC=~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
RS="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

say "Target: $HOST port $PORT  ->  ~/qwen-0.8b/wsp   (WSP pod, not the insurance pod)"
say "Fable 5.1 · $BUDGET passes · cap \$$CAP · arm name: fable_clean"

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"
say "Pod reachable."

say "1/6  Checking the pod is not already busy"
# Only IN-FLIGHT states count. EXIT_n / ABORTED / FINISHED are leftovers from a previous run
# and must never block a relaunch - that bug cost an evening on its own.
BUSY=$($SSH "$HOST" '
  live() { pgrep -f "$1" >/dev/null 2>&1; }
  live "[e]ditor_api\.py --task"  && echo EDITOR
  live "[s]erver\.py --budget"    && echo SERVER
  live "[e]val_heldout\.py --arm" && echo EVAL
  for pf in ~/qwen-0.8b/wsp/work/server.pid ~/qwen-4b/wsp/work/server.pid; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "PIDFILE:$pf"
  done
  for sf in ~/qwen-0.8b/POD_QUEUE_STATUS ~/qwen-0.8b/COMBOS_STATUS ~/qwen-0.8b/REMEASURE_STATUS ~/qwen-0.8b/FABLE_CLEAN_STATUS; do
    st=$(cat "$sf" 2>/dev/null)
    case "$st" in A_CONTEXT_ARMS|B_REMEASURE|C_FABLE_CLEAN|RUNNING) echo "$(basename $sf):$st" ;; esac
  done
  true' < /dev/null)
if [ -n "$BUSY" ]; then
  printf '\n\033[33mWARNING: the pod looks busy: %s\033[0m\n' "$(echo $BUSY | tr '\n' ' ')"
  echo "This run costs real money and would share one GPU with whatever is already there."
  echo "Double-click status.command in another window to see what is actually live."
  printf "Type 'yes' to launch anyway, anything else to abort: "
  read ANS; [ "$ANS" = yes ] || die "aborted - nothing was started, nothing was spent"
fi
say "Not busy."

say "2/6  Sending anything the pod is missing (this is the step that used to just give up)"
W=$($SSH "$HOST" 'ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l' < /dev/null)
if [ "${W:-0}" -eq 0 ]; then
  say "   The pod has no 0.8B weights - its disk is ephemeral, so a restart loses them."
  say "   Sending ~1.6 GB. This is the slow step, a few minutes."
  rsync -az -e "$RS" --include '*/' --include '*.safetensors' --include '*.json' \
      --include 'tokenizer*' --exclude '*' "$SRC/../" "$HOST:~/qwen-0.8b/" \
      || die "could not send the 0.8B weights"
  say "   Weights sent."
else
  echo "     weights already on the pod ($W files)"
fi
for f in run_compare.sh server.py eval_heldout.py envfile.py; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/$f" "$HOST:~/qwen-0.8b/wsp/$f" || die "could not copy $f"
done
scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/fable_08b_clean.sh" "$HOST:~/qwen-0.8b/wsp/gpu/" || die "could not copy the runner"
$SSH "$HOST" 'chmod +x ~/qwen-0.8b/wsp/gpu/fable_08b_clean.sh; mkdir -p ~/qwen-0.8b/wsp/work/best_paper ~/qwen-0.8b/wsp/contexts ~/qwen-0.8b/wsp/tasks ~/qwen-0.8b/wsp/eval' < /dev/null
rsync -az -e "$RS" "$SRC/work/best_paper/" "$HOST:~/qwen-0.8b/wsp/work/best_paper/" || die "could not send the loop-fix seed"
rsync -az -e "$RS" "$SRC/tasks/"           "$HOST:~/qwen-0.8b/wsp/tasks/"           || die "could not send tasks/"
rsync -az -e "$RS" "$SRC/contexts/"        "$HOST:~/qwen-0.8b/wsp/contexts/"        || die "could not send contexts/"
# the runner seals these out of the editor's tree itself, but they must BE there to be sealed
rsync -az --delete -e "$RS" "$SRC/eval/heldout"  "$HOST:~/qwen-0.8b/wsp/eval/" 2>/dev/null \
  || echo "     note: eval/heldout not sent (it may already be sealed on the pod - the runner handles that)"
rsync -az --delete -e "$RS" "$SRC/practice/"     "$HOST:~/qwen-0.8b/wsp/practice/" || die "could not send practice/"
say "Harness, runner, seed, briefs and context are on the pod."

say "3/6  Narrowing the pod's .env to the one key this run needs"
# push.sh copied the whole .env, so all three keys are sitting on the pod. This run uses Fable only.
grep '^ANTHROPIC_API_KEY=' "$SRC/.env" \
  | $SSH "$HOST" 'umask 077; cat > ~/qwen-0.8b/wsp/.env' \
  || die "could not write the key to the pod"
say "Pod .env now holds ONLY ANTHROPIC_API_KEY, mode 600. (Your OpenAI and Fireworks keys are off that box.)"

say "4/6  Verifying - each line stands on its own, so a failure names itself"
REPORT=$($SSH "$HOST" 'cd ~/qwen-0.8b/wsp 2>/dev/null || { echo "FAIL no ~/qwen-0.8b/wsp"; exit; }
ok () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "FAIL $2"; fi; }
ok "$(grep -c "^ANTHROPIC_API_KEY=.\{20,\}" .env 2>/dev/null)"           "ANTHROPIC_API_KEY present in .env"
ok "$(grep -c "held-out ids are not available to the editor" server.py)"  "leak guard in server.py"
ok "$(grep -c WSP_EVAL_DIR server.py)"                                    "WSP_EVAL_DIR patch in server.py"
ok "$(grep -c WSP_EVAL_DIR eval_heldout.py)"                              "WSP_EVAL_DIR patch in eval_heldout.py"
ok "$(grep -c fable_clean run_compare.sh)"                                "fable_clean route in run_compare.sh"
ok "$(ls gpu/fable_08b_clean.sh 2>/dev/null | wc -l)"                     "gpu/fable_08b_clean.sh"
ok "$(ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l)"                  "0.8B weights in ~/qwen-0.8b"
ok "$(ls work/best_paper/*.npy 2>/dev/null | wc -l)"                      "work/best_paper loop-fix seed (.npy files)"
ok "$(ls contexts/advising.md 2>/dev/null | wc -l)"                       "contexts/advising.md"
ok "$(ls tasks/*.md tasks/*.txt 2>/dev/null | wc -l)"                     "tasks/ brief"
if [ -x .venv/bin/python ] || [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then echo "ok   a usable venv"
else echo "FAIL a usable venv (neither wsp/.venv nor ~/qwen-4b/wsp/.venv)"; fi
STRAY=$(ls practice/ 2>/dev/null | grep -c heldout)
if [ "${STRAY:-0}" -eq 0 ]; then echo "ok   no held-out file left in practice/"
else echo "FAIL practice/ still holds $STRAY held-out file(s)"; fi' < /dev/null)
echo "$REPORT" | sed 's/^/     /'
if echo "$REPORT" | grep -q FAIL; then
  die "the line(s) marked FAIL above are what is missing. Nothing launched, nothing spent."
fi
say "All twelve checks passed."

say "5/6  Clearing any leftover server or editor on the pod"
$SSH "$HOST" 'pkill -f "server.py --budget" 2>/dev/null; pkill -f "editor_api.py --task" 2>/dev/null; sleep 2; true' < /dev/null

say "6/6  Launching, detached from this terminal and from the Mac"
$SSH "$HOST" "cd ~/qwen-0.8b/wsp && BUDGET=$BUDGET WSP_MAX_COST_USD=$CAP setsid nohup bash gpu/fable_08b_clean.sh > ~/qwen-0.8b/fable_clean_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 15
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/FABLE_CLEAN_STATUS 2>/dev/null; echo "---"; tail -40 ~/qwen-0.8b/fable_clean_*.log 2>/dev/null | tail -40' < /dev/null

cat <<'TAIL'

=========================================================
Read the status block above before you close this window.

  C_FABLE_CLEAN or RUNNING   -> it is working. Close the Mac.
  ABORTED + an ABORT: line   -> it stopped itself; the line says why.
  nothing at all             -> the runner did not start; send me
                                the whole window.

Check on it later:
  double-click  status.command

DO NOT stop the pod in the RunPod console: its disk is ephemeral
and stopping it erases the run.
=========================================================
TAIL
echo "Press return to close."; read _
