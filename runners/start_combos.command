#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Measures the edit+context combos, and the LoRA, on the
# provenance-clean replacement set. No editor, no API spend. Roughly 2-3 hours on the pod.
# Once it reports "launched" the run lives on the POD: close this window, close the Mac.
#
# Fills the gap in the clean numbers: gpu/remeasure_v2.sh measured nine single edits, the
# base and context alone, and no *_context combo at all. So sonnet_context and the other
# six combos have numbers ONLY from the compromised set, which is not comparable.
#
# THIS IS THE WEIGHT-SPACE-PROGRAMMING POD. It is NOT the insurance RSI pod.
HOST=root@69.8.146.173
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SRC=~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

say "Target: $HOST port $PORT -> ~/qwen-0.8b/wsp   (WSP pod, not the insurance pod)"
say "Arms: sonnet/fable/opus/astra/astra80/astra_xhigh/kimi + context, and the LoRA. No API spend."

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"

say "1/4  Checking the pod is not already busy"
# This matters more here than usual: evaluation loads its own copy of the model and would
# contend for the GPU with a live editor arm, and every phase writes to the shared edited/
# directory. Only IN-FLIGHT states count. EXIT_n / ABORTED / FINISHED are leftovers.
BUSY=$($SSH "$HOST" '
  live() { pgrep -f "$1" >/dev/null 2>&1; }
  live "[e]ditor_api\.py --task" && echo EDITOR
  live "[s]erver\.py --budget"   && echo SERVER
  for pf in ~/qwen-0.8b/wsp/work/server.pid ~/qwen-4b/wsp/work/server.pid; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "PIDFILE:$pf"
  done
  for sf in ~/qwen-0.8b/POD_QUEUE_STATUS ~/qwen-0.8b/FABLE_CLEAN_STATUS ~/qwen-0.8b/REMEASURE_STATUS ~/qwen-0.8b/COMBOS_STATUS; do
    st=$(cat "$sf" 2>/dev/null)
    case "$st" in
      A_CONTEXT_ARMS|B_REMEASURE|C_FABLE_CLEAN|RUNNING) echo "$(basename $sf):$st" ;;
    esac
  done
  true' < /dev/null)
if [ -n "$BUSY" ]; then
  printf '\n\033[33mWARNING: the pod looks busy: %s\033[0m\n' "$(echo $BUSY | tr '\n' ' ')"
  echo "Two things would collide: both jobs load the model onto one GPU, and both write to"
  echo "the shared edited/ directory. If the clean Fable arm really is running, let it finish."
  echo "Double-click status.command in another window if you want to see what is actually live."
  printf "Type 'yes' to run anyway, anything else to abort: "
  read ANS; [ "$ANS" = yes ] || die "aborted - nothing was started"
fi

say "2/4  Pushing the runner and the patched harness"
scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/remeasure_combos.sh" "$HOST:~/qwen-0.8b/wsp/gpu/" \
  || die "could not copy gpu/remeasure_combos.sh. Is it in $SRC/gpu/ on this Mac?"
for f in eval_heldout.py server.py envfile.py; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/$f" "$HOST:~/qwen-0.8b/wsp/$f" || die "could not push $f"
done
$SSH "$HOST" 'chmod +x ~/qwen-0.8b/wsp/gpu/remeasure_combos.sh'
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  "$SRC/contexts/" "$HOST:~/qwen-0.8b/wsp/contexts/" || die "could not push contexts/"
say "Pushed."

say "3/4  Verifying what the pod has (each line stands on its own)"
REPORT=$($SSH "$HOST" 'cd ~/qwen-0.8b/wsp 2>/dev/null || { echo "FAIL no ~/qwen-0.8b/wsp"; exit; }
ok () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "FAIL $2"; fi; }
note () { if [ "$1" -gt 0 ] 2>/dev/null; then echo "ok   $2"; else echo "note $2"; fi; }
ok "$(grep -c WSP_EVAL_DIR eval_heldout.py)"                      "WSP_EVAL_DIR patch in eval_heldout.py"
ok "$(grep -c WSP_BASE_ARM eval_heldout.py)"                      "WSP_BASE_ARM patch in eval_heldout.py"
ok "$(ls gpu/remeasure_combos.sh 2>/dev/null | wc -l)"            "gpu/remeasure_combos.sh"
ok "$(ls contexts/advising.md 2>/dev/null | wc -l)"               "contexts/advising.md"
ok "$(ls ~/qwen-0.8b/*.safetensors 2>/dev/null | wc -l)"          "0.8B weights in ~/qwen-0.8b"
ok "$(ls -d work/eval_v2_base 2>/dev/null | wc -l)"               "work/eval_v2_base (the clean base row to take deltas against)"
H2=0
for c in eval/heldout2 /opt/wsp_v2/eval/heldout /opt/wsp_sealed_08b/eval/heldout2; do
  ls $c/*.jsonl >/dev/null 2>&1 && grep -ql h2_ $c/*.jsonl 2>/dev/null && { echo "ok   replacement set found at $c"; H2=1; break; }
done
[ "$H2" = 1 ] || echo "FAIL the replacement set is nowhere on this box (eval/heldout2, /opt/wsp_v2, or the sealed copy)"
for a in sonnet fable opus astra astra80 astra_xhigh kimi; do
  n=$(ls runs/full_$a/edited_final/*.npy 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && [ "$a" = sonnet ] && n=$(ls runs/full_sonnet/edited_partial/*.npy 2>/dev/null | wc -l)
  note "$n" "frozen edit for ${a}_context ($n tensors)"
done
note "$(ls runs/full_finetune/edited_final/*.npy 2>/dev/null | wc -l)" "LoRA as a dense diff (0 = left unmeasured, not an error)"' < /dev/null)
echo "$REPORT" | sed 's/^/     /'
if echo "$REPORT" | grep -q FAIL; then
  die "the pod failed the checks marked FAIL above. Nothing launched."
fi
say "Everything the run needs is present. Lines marked 'note' are arms that will be skipped, not errors."

say "4/4  Launching, detached from this terminal and from the Mac"
$SSH "$HOST" "cd ~/qwen-0.8b/wsp && setsid nohup bash gpu/remeasure_combos.sh > ~/qwen-0.8b/combos_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 12
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/COMBOS_STATUS 2>/dev/null; echo "---"; tail -30 ~/qwen-0.8b/remeasure_combos_*.log 2>/dev/null | tail -30' < /dev/null

cat <<'TAIL'

=========================================================
Running on the pod. You can close the Mac. No API spend.
Expect 2-3 hours: the 1.3k-token advising context makes each
context arm slower than a bare eval.

Check on it:
  double-click  status.command
or
  ssh -i ~/qwen-0.8b/.ssh_runpod/id_ed25519 -p 11873 root@69.8.146.173 \
    'cat ~/qwen-0.8b/COMBOS_STATUS; tail -60 ~/qwen-0.8b/remeasure_combos_*.log'

It ends by printing each single edit beside its edit+context
combo, all on the clean set, so the question "does context add
anything on top of the edit?" gets a straight answer.

DO NOT stop the pod in the RunPod console: its disk is ephemeral.
=========================================================
TAIL
echo "Press return to close."; read _
