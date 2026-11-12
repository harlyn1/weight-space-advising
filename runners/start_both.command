#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Runs everything that is still owed, in one go:
#
#   0. pulls the clean re-measurement OFF the pod first   <-- the important part
#   1. the edit + context combos on the clean set   ~2-3 h   no API spend
#   2. the clean Fable editor arm                   ~17-20 h capped at $CAP
#
# They run in sequence, not together: both load the model onto the one GPU.
# Once it reports "launched" the work lives on the POD. Close this window and the Mac.
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
say "Phase 1 combos: free. Phase 2 Fable: capped at \$$CAP, $BUDGET passes."

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"
say "Pod reachable."

say "1/7  Checking the pod is not already busy"
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
  for sf in ~/qwen-0.8b/BOTH_STATUS ~/qwen-0.8b/POD_QUEUE_STATUS ~/qwen-0.8b/COMBOS_STATUS \
            ~/qwen-0.8b/REMEASURE_STATUS ~/qwen-0.8b/FABLE_CLEAN_STATUS; do
    st=$(cat "$sf" 2>/dev/null)
    case "$st" in P0_BASE_CHECK|P1_COMBOS|P2_FABLE_CLEAN|A_CONTEXT_ARMS|B_REMEASURE|C_FABLE_CLEAN|RUNNING)
      echo "$(basename $sf):$st" ;; esac
  done
  true' < /dev/null)
if [ -n "$BUSY" ]; then
  printf '\n\033[33mWARNING: the pod looks busy: %s\033[0m\n' "$(echo $BUSY | tr '\n' ' ')"
  echo "Everything here shares one GPU, and phase 2 costs real money."
  echo "Double-click status.command in another window to see what is actually live."
  printf "Type 'yes' to launch anyway, anything else to abort: "
  read ANS; [ "$ANS" = yes ] || die "aborted - nothing was started, nothing was spent"
fi
say "Not busy."

say "2/7  PULLING the clean re-measurement off the pod BEFORE anything else"
# Those ten arms exist only on the pod's ephemeral disk. If it stops, the source data behind
# every published number is gone. This costs seconds and runs before any risk is taken.
mkdir -p "$SRC/work"
if rsync -az -e "$RS" "$HOST:~/qwen-0.8b/wsp/work/eval_v2_*" "$SRC/work/" 2>/dev/null; then
  N=$(ls -d "$SRC"/work/eval_v2_* 2>/dev/null | wc -l | tr -d ' ')
  say "   Pulled $N clean result directories to this Mac. They are now safe from the pod."
  ls -d "$SRC"/work/eval_v2_* 2>/dev/null | sed 's|.*/|     |'
else
  printf '\n\033[33m   Nothing pulled. Either the pod restarted and lost them, or they were never there.\033[0m\n'
  echo "   Not fatal: phase 0 on the pod rebuilds the clean base row if it is missing"
  echo "   (one eval, ~11 min, no API spend). But the other nine arms would be gone."
fi

say "3/7  Sending anything the pod is missing"
W=$($SSH "$HOST" 'ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l' < /dev/null)
if [ "${W:-0}" -eq 0 ]; then
  say "   The pod has no 0.8B weights. Sending ~1.6 GB - the slow step, a few minutes."
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
for f in queue_both.sh remeasure_combos.sh fable_08b_clean.sh; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/$f" "$HOST:~/qwen-0.8b/wsp/gpu/$f" || die "could not copy gpu/$f"
done
$SSH "$HOST" 'chmod +x ~/qwen-0.8b/wsp/gpu/queue_both.sh ~/qwen-0.8b/wsp/gpu/remeasure_combos.sh ~/qwen-0.8b/wsp/gpu/fable_08b_clean.sh
  mkdir -p ~/qwen-0.8b/wsp/work/best_paper ~/qwen-0.8b/wsp/contexts ~/qwen-0.8b/wsp/tasks ~/qwen-0.8b/wsp/eval' < /dev/null
rsync -az -e "$RS" "$SRC/work/best_paper/" "$HOST:~/qwen-0.8b/wsp/work/best_paper/" || die "could not send the loop-fix seed"
rsync -az -e "$RS" "$SRC/tasks/"           "$HOST:~/qwen-0.8b/wsp/tasks/"           || die "could not send tasks/"
rsync -az -e "$RS" "$SRC/contexts/"        "$HOST:~/qwen-0.8b/wsp/contexts/"        || die "could not send contexts/"
rsync -az --delete -e "$RS" "$SRC/practice/"      "$HOST:~/qwen-0.8b/wsp/practice/"  || die "could not send practice/"
rsync -az --delete -e "$RS" "$SRC/eval/heldout"   "$HOST:~/qwen-0.8b/wsp/eval/" 2>/dev/null \
  || echo "     note: eval/heldout not sent (already sealed on the pod - the runner handles that)"
rsync -az --delete -e "$RS" "$SRC/eval/heldout2"  "$HOST:~/qwen-0.8b/wsp/eval/"      || die "could not send eval/heldout2"
STRAY=$($SSH "$HOST" 'ls ~/qwen-0.8b/wsp/practice/ 2>/dev/null | grep -c heldout' < /dev/null)
[ "${STRAY:-0}" -eq 0 ] || die "the pod's practice/ still holds $STRAY held-out file(s) after the push - stopping rather than letting an editor reach them"
say "Harness, runners, seed, briefs, context and both eval sets are on the pod."

say "4/7  Building the LoRA's edit snapshot, then sending every frozen edit"
# The LoRA is not adapter weights needing a loader - it is a fully merged 1.5 GB copy at
# finetune/merged, and finetune/merged_to_edits.py turns it into the harness's own .npy
# format (the 96 LoRA-touched projections). That runs HERE, on the Mac, so the 1.5 GB
# never crosses the wire; only the changed tensors do.
if ls "$SRC"/runs/full_finetune/edited_final/*.npy >/dev/null 2>&1; then
  echo "     LoRA snapshot already built ($(ls "$SRC"/runs/full_finetune/edited_final/*.npy | wc -l | tr -d ' ') tensors)"
elif [ -d "$SRC/finetune/merged" ]; then
  LPY="$SRC/.venv/bin/python"; [ -x "$LPY" ] || LPY=python3
  echo "     building it from finetune/merged - a few minutes, it compares every tensor"
  # edited/ is shared scratch and currently holds another arm's tensors. merged_to_edits.py
  # only copies the names it wrote, so the snapshot would be correct either way, but a
  # stale edited/ has bitten this project before. Clear it.
  mkdir -p "$SRC/edited" && rm -f "$SRC"/edited/*.npy
  if ( cd "$SRC" && "$LPY" finetune/merged_to_edits.py ); then
    echo "     LoRA snapshot built: $(ls "$SRC"/runs/full_finetune/edited_final/*.npy 2>/dev/null | wc -l | tr -d ' ') tensors"
  else
    printf '\033[33m     merged_to_edits.py failed - continuing WITHOUT the LoRA arm.\033[0m\n'
    echo "     Everything else still runs. Send me the lines above and I will fix it."
  fi
  rm -f "$SRC"/edited/*.npy
else
  echo "     no finetune/merged on this Mac - the LoRA stays unmeasured"
fi
echo
for a in sonnet fable opus astra astra80 astra_xhigh kimi finetune; do
  D="$SRC/runs/full_$a/edited_final"
  [ -d "$D" ] || { echo "     $a: no edited_final on this Mac - that combo will be skipped"; continue; }
  $SSH "$HOST" "mkdir -p ~/qwen-0.8b/wsp/runs/full_$a" < /dev/null
  rsync -az -e "$RS" "$D" "$HOST:~/qwen-0.8b/wsp/runs/full_$a/" 2>/dev/null \
    && echo "     $a  ($(ls "$D"/*.npy 2>/dev/null | wc -l | tr -d ' ') tensors)" || echo "     $a  (failed)"
done

say "5/7  Narrowing the pod's .env to the one key this needs"
grep '^ANTHROPIC_API_KEY=' "$SRC/.env" \
  | $SSH "$HOST" 'umask 077; cat > ~/qwen-0.8b/wsp/.env' \
  || die "could not write the key to the pod"

say "6/7  Verifying - each line stands on its own, so a failure names itself"
REPORT=$($SSH "$HOST" 'cd ~/qwen-0.8b/wsp 2>/dev/null || { echo "FAIL no ~/qwen-0.8b/wsp"; exit; }
ok () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "FAIL $2"; fi; }
note () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "note $2"; fi; }
ok "$(grep -c "^ANTHROPIC_API_KEY=.\{20,\}" .env 2>/dev/null)"           "ANTHROPIC_API_KEY present in .env"
ok "$(grep -c "held-out ids are not available to the editor" server.py)"  "leak guard in server.py"
ok "$(grep -c WSP_EVAL_DIR server.py)"                                    "WSP_EVAL_DIR patch in server.py"
ok "$(grep -c WSP_EVAL_DIR eval_heldout.py)"                              "WSP_EVAL_DIR patch in eval_heldout.py"
ok "$(grep -c WSP_BASE_ARM eval_heldout.py)"                              "WSP_BASE_ARM patch in eval_heldout.py"
ok "$(grep -c fable_clean run_compare.sh)"                                "fable_clean route in run_compare.sh"
ok "$(ls gpu/queue_both.sh gpu/remeasure_combos.sh gpu/fable_08b_clean.sh 2>/dev/null | wc -l)" "all three runners present"
ok "$(ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l)"                  "0.8B weights in ~/qwen-0.8b"
ok "$(ls work/best_paper/*.npy 2>/dev/null | wc -l)"                      "work/best_paper loop-fix seed"
ok "$(ls contexts/advising.md 2>/dev/null | wc -l)"                       "contexts/advising.md"
ok "$(ls tasks/*.md tasks/*.txt 2>/dev/null | wc -l)"                     "tasks/ brief"
ok "$(ls eval/heldout2/*.jsonl 2>/dev/null | wc -l)"                      "eval/heldout2 (the replacement set)"
if [ -x .venv/bin/python ] || [ -x ~/qwen-4b/wsp/.venv/bin/python ]; then echo "ok   a usable venv"
else echo "FAIL a usable venv (neither wsp/.venv nor ~/qwen-4b/wsp/.venv)"; fi
note "$(ls -d work/eval_v2_base 2>/dev/null | wc -l)"                     "work/eval_v2_base (absent = phase 0 rebuilds it, ~11 min)"
for a in sonnet fable opus astra astra80 astra_xhigh kimi; do
  note "$(ls runs/full_$a/edited_final/*.npy 2>/dev/null | wc -l)"        "frozen edit for ${a}_context"
done' < /dev/null)
echo "$REPORT" | sed 's/^/     /'
if echo "$REPORT" | grep -q FAIL; then
  die "the line(s) marked FAIL above are what is missing. Nothing launched, nothing spent."
fi
say "Every required check passed. Lines marked 'note' are arms that get skipped, not errors."

say "7/7  Launching the queue, detached from this terminal and from the Mac"
$SSH "$HOST" 'pkill -f "server.py --budget" 2>/dev/null; pkill -f "editor_api.py --task" 2>/dev/null; sleep 2; true' < /dev/null
$SSH "$HOST" "cd ~/qwen-0.8b/wsp && BUDGET=$BUDGET WSP_MAX_COST_USD=$CAP setsid nohup bash gpu/queue_both.sh > ~/qwen-0.8b/queue_both_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 15
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/BOTH_STATUS 2>/dev/null; echo "---"; tail -40 ~/qwen-0.8b/queue_both_*.log 2>/dev/null | tail -40' < /dev/null

cat <<'TAIL'

=========================================================
Read the status block above before you close this window.

  P0_BASE_CHECK / P1_COMBOS / P2_FABLE_CLEAN -> working
  ABORTED + an ABORT: line  -> it stopped itself; that line says why
  nothing at all            -> the runner never started; send me
                               the whole window

Status reads: P0_BASE_CHECK -> P1_COMBOS -> P2_FABLE_CLEAN -> FINISHED
Combos land in 2-3 h. Fable takes most of a day after that, spend-capped,
so it can be stopped on whatever the combos turn out to show.

Check on it:
  double-click  status.command

DO NOT stop the pod in the RunPod console: its disk is ephemeral.
=========================================================
TAIL
echo "Press return to close."; read _
