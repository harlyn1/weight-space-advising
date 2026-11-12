#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Queues all remaining work on the pod, in one launch:
#   A. the three missing context arms      ~1-2 h   no API spend
#   B. the heldout2 re-measurement         ~2 h     no API spend
#   C. the clean Fable editor arm          ~17-20 h capped at $CAP
# Once it reports "launched" everything lives on the POD. Close this window and the Mac.
#
# THIS IS THE WEIGHT-SPACE-PROGRAMMING POD. It is NOT the insurance RSI pod.
HOST=root@69.8.146.173
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
CAP=${CAP:-250}
BUDGET=${BUDGET:-40}
SRC=~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
RS="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

say "Target: $HOST port $PORT -> ~/qwen-0.8b/wsp   (WSP pod, not the insurance pod)"
say "Phases A+B cost nothing in API. Phase C is Fable, capped at \$$CAP."

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"

say "1/5  Checking the pod is not already busy"
BUSY=$($SSH "$HOST" '
  live() { pgrep -f "$1" >/dev/null 2>&1; }
  live "[e]ditor_api\.py --task" && echo EDITOR
  live "[s]erver\.py --budget"   && echo SERVER
  for pf in ~/qwen-0.8b/wsp/work/server.pid ~/qwen-4b/wsp/work/server.pid; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "PIDFILE:$pf"
  done
  # only an IN-FLIGHT status counts as busy. EXIT_n / ABORTED / FINISHED are terminal
  # states left behind by a previous run and must not block a relaunch.
  st=$(cat ~/qwen-0.8b/POD_QUEUE_STATUS 2>/dev/null)
  case "$st" in
    A_CONTEXT_ARMS|B_REMEASURE|C_FABLE_CLEAN|RUNNING) echo "STATUS:$st" ;;
  esac
  true' < /dev/null)
if [ -n "$BUSY" ]; then
  printf '\n\033[33mWARNING: the pod looks busy: %s\033[0m\n' "$(echo $BUSY | tr '\n' ' ')"
  echo "Every phase writes to the shared edited/ directory, so two queues would corrupt each other."
  printf "Type 'yes' to continue anyway, anything else to abort: "
  read ANS; [ "$ANS" = yes ] || die "aborted - nothing was started"
fi

say "2/5  Pushing everything the pod needs (repairing, not just checking)"
# Anything the pod is missing is sent from here rather than aborting. rsync moves only deltas.
MISSING_W=$($SSH "$HOST" 'ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l' < /dev/null)
if [ "${MISSING_W:-0}" -eq 0 ]; then
  say "   The pod has no 0.8B weights. Sending ~1.6 GB - this is the slow step, a few minutes."
  rsync -az --progress -e "$RS" --include '*/' --include '*.safetensors' --include '*.json' \
      --include 'tokenizer*' --exclude '*' "$SRC/../" "$HOST:~/qwen-0.8b/" \
      || die "could not send the 0.8B weights"
  say "   Weights sent."
fi
for f in run_compare.sh server.py eval_heldout.py envfile.py; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/$f" "$HOST:~/qwen-0.8b/wsp/$f" || die "could not push $f"
done
for f in pod_queue.sh remeasure_v2.sh fable_08b_clean.sh; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/$f" "$HOST:~/qwen-0.8b/wsp/gpu/$f" || die "could not push $f"
done
# the small trees the phases read from: briefs, the advising context, the loop-fix seed
$SSH "$HOST" 'mkdir -p ~/qwen-0.8b/wsp/work/best_paper ~/qwen-0.8b/wsp/contexts ~/qwen-0.8b/wsp/tasks' < /dev/null
rsync -az -e "$RS" "$SRC/tasks/"           "$HOST:~/qwen-0.8b/wsp/tasks/"           || die "could not push tasks/"
rsync -az -e "$RS" "$SRC/contexts/"        "$HOST:~/qwen-0.8b/wsp/contexts/"        || die "could not push contexts/"
rsync -az -e "$RS" "$SRC/work/best_paper/" "$HOST:~/qwen-0.8b/wsp/work/best_paper/" || die "could not push the loop-fix seed"
$SSH "$HOST" 'chmod +x ~/qwen-0.8b/wsp/gpu/pod_queue.sh ~/qwen-0.8b/wsp/gpu/remeasure_v2.sh ~/qwen-0.8b/wsp/gpu/fable_08b_clean.sh'
$SSH "$HOST" 'mkdir -p ~/qwen-0.8b/wsp/eval' < /dev/null
# BOTH eval sets and practice/: the 2026-11-10 leak fix MOVED the held-out files out of
# practice/ and into eval/heldout/, so a pod pushed before that still has the old layout.
rsync -az --delete -e "$RS" "$SRC/eval/heldout"  "$HOST:~/qwen-0.8b/wsp/eval/" || die "could not push eval/heldout"
rsync -az --delete -e "$RS" "$SRC/eval/heldout2" "$HOST:~/qwen-0.8b/wsp/eval/" || die "could not push eval/heldout2"
rsync -az --delete -e "$RS" "$SRC/practice/"     "$HOST:~/qwen-0.8b/wsp/practice/" || die "could not push practice/"
say "Harness, runners, both eval sets and the corrected practice/ are on the pod."
STRAY=$($SSH "$HOST" 'ls ~/qwen-0.8b/wsp/practice/ 2>/dev/null | grep -c heldout' < /dev/null)
[ "${STRAY:-0}" -eq 0 ] || die "the pod's practice/ still holds $STRAY held-out file(s) after the push - stopping rather than letting an editor reach them"
say "Confirmed: no held-out file remains in the pod's practice/."

say "3/5  Pushing the frozen edits (rsync sends only what the pod is missing)"
for a in astra_published fable opus astra astra80 astra_xhigh kimi sonnet; do
  D="$SRC/runs/full_$a/edited_final"
  [ -d "$D" ] || { echo "     $a: no edited_final here - skipped"; continue; }
  $SSH "$HOST" "mkdir -p ~/qwen-0.8b/wsp/runs/full_$a" < /dev/null
  rsync -az -e "$RS" "$D" "$HOST:~/qwen-0.8b/wsp/runs/full_$a/" 2>/dev/null \
    && echo "     $a  ($(du -sh "$D" | cut -f1))" || echo "     $a  (failed)"
done

say "4/5  Narrowing the pod's .env to the one key this needs"
# push.sh copied the whole .env, so all three keys are on that box. Only Fable is used here.
grep '^ANTHROPIC_API_KEY=' "$SRC/.env" \
  | $SSH "$HOST" 'umask 077; cat > ~/qwen-0.8b/wsp/.env' \
  || die "could not write the key to the pod"
REPORT=$($SSH "$HOST" 'cd ~/qwen-0.8b/wsp 2>/dev/null || { echo "FAIL no ~/qwen-0.8b/wsp"; exit; }
ok () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "FAIL $2"; fi; }
ok "$(grep -c "^ANTHROPIC_API_KEY=.\{20,\}" .env 2>/dev/null)"            "ANTHROPIC_API_KEY present in .env"
ok "$(grep -c "held-out ids are not available to the editor" server.py)"   "leak guard in server.py"
ok "$(grep -c WSP_EVAL_DIR server.py)"                                     "WSP_EVAL_DIR patch in server.py"
ok "$(grep -c WSP_EVAL_DIR eval_heldout.py)"                               "WSP_EVAL_DIR patch in eval_heldout.py"
ok "$(grep -c fable_clean run_compare.sh)"                                 "fable_clean route in run_compare.sh"
ok "$(ls -d eval/heldout  2>/dev/null | wc -l)"                            "eval/heldout present"
ok "$(ls -d eval/heldout2 2>/dev/null | wc -l)"                            "eval/heldout2 present"
ok "$(ls work/best_paper/*.npy 2>/dev/null | wc -l)"                       "work/best_paper loop-fix seed"
ok "$(ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l)"                   "0.8B weights in ~/qwen-0.8b"
ok "$(ls contexts/advising.md 2>/dev/null | wc -l)"                        "contexts/advising.md"' < /dev/null)
echo "$REPORT" | sed 's/^/     /'
if echo "$REPORT" | grep -q FAIL; then
  die "the pod failed the checks marked FAIL above. Nothing launched."
fi
say "All checks passed. Pod .env holds ONLY ANTHROPIC_API_KEY, mode 600."

say "5/5  Launching the queue, detached from this terminal and from the Mac"
$SSH "$HOST" "cd ~/qwen-0.8b/wsp && BUDGET=$BUDGET WSP_MAX_COST_USD=$CAP setsid nohup bash gpu/pod_queue.sh > ~/qwen-0.8b/pod_queue_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 12
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/POD_QUEUE_STATUS 2>/dev/null; echo "---"; tail -30 ~/qwen-0.8b/pod_queue_*.log 2>/dev/null | tail -30' < /dev/null

cat <<'TAIL'

=========================================================
The whole queue is on the pod. You can close the Mac.

Status reads: A_CONTEXT_ARMS -> B_REMEASURE -> C_FABLE_CLEAN -> FINISHED

Check on it:
  ssh -i ~/qwen-0.8b/.ssh_runpod/id_ed25519 -p 11873 root@69.8.146.173 \
    'cat ~/qwen-0.8b/POD_QUEUE_STATUS; tail -60 ~/qwen-0.8b/pod_queue_*.log'

Pull results:
  double-click  ~/qwen-0.8b/wsp/gpu/pull_clean.command

By morning expect A and B finished and C running. C is spend-capped,
so it can be stopped on whatever A and B turn out to show.

DO NOT stop the pod in the RunPod console: its disk is ephemeral.
=========================================================
TAIL
echo "Press return to close."; read _
