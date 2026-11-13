#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Re-measures every frozen edit on the 141 replacement items.
# No editor, no API spend - pure evaluation, roughly 1-2 hours on the pod.
# Once it reports "launched" the run lives on the POD: close this window and the Mac.
#
# THIS IS THE WEIGHT-SPACE-PROGRAMMING POD. It is NOT the insurance RSI pod.
HOST=root@POD_IP
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SRC=~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
RS="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

say "Target: $HOST port $PORT -> ~/qwen-0.8b/wsp   (WSP pod, not the insurance pod)"
say "Re-measuring every frozen edit on the replacement set. No API spend."

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"

say "1/4  Checking nothing else is running on that pod"
BUSY=$($SSH "$HOST" '
  live() { pgrep -f "$1" >/dev/null 2>&1; }
  live "[e]ditor_api\.py --task" && echo EDITOR
  live "[s]erver\.py --budget"   && echo SERVER
  for pf in ~/qwen-0.8b/wsp/work/server.pid ~/qwen-4b/wsp/work/server.pid; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null)
    [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "PIDFILE:$pf"
  done
  cat ~/qwen-0.8b/POD_QUEUE_STATUS 2>/dev/null
  true' < /dev/null)
if [ -n "$BUSY" ]; then
  printf '\n\033[33mWARNING: the pod is busy (%s).\033[0m\n' "$(echo $BUSY | tr '\n' ' ')"
  echo "Evaluation loads its own model and would contend for the GPU, and stop_server kills by"
  echo "pattern across folders, so the two could interfere."
  printf "Type 'yes' to continue anyway, anything else to abort: "
  read ANS; [ "$ANS" = yes ] || die "aborted - nothing was started"
fi

say "2/4  Pushing the replacement set, the frozen edits and the patched harness"
rsync -az -e "$RS" "$SRC/eval/heldout2" "$HOST:~/qwen-0.8b/wsp/eval/" || die "could not push heldout2"
echo "  pushed eval/heldout2"
for f in eval_heldout.py server.py envfile.py; do
  scp -q -i "$KEY" -P "$PORT" "$SRC/$f" "$HOST:~/qwen-0.8b/wsp/$f" || die "could not push $f"
done
echo "  pushed the patched harness"
scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/remeasure_v2.sh" "$HOST:~/qwen-0.8b/wsp/gpu/" || die "could not push the runner"
$SSH "$HOST" 'chmod +x ~/qwen-0.8b/wsp/gpu/remeasure_v2.sh'

say "   frozen edits (only what the pod is missing is sent):"
# only the submitted edits: edited_partial would double the transfer for no gain
for a in astra_published fable opus astra astra80 astra_xhigh kimi sonnet finetune; do
  D="$SRC/runs/full_$a/edited_final"
  [ -d "$D" ] || { echo "     $a: no edited_final on this Mac - skipped"; continue; }
  $SSH "$HOST" "mkdir -p ~/qwen-0.8b/wsp/runs/full_$a" < /dev/null
  rsync -az -e "$RS" "$D" "$HOST:~/qwen-0.8b/wsp/runs/full_$a/" 2>/dev/null \
    && echo "     $a  ($(du -sh "$D" | cut -f1))" || echo "     $a  (failed)"
done

say "3/4  Verifying the pod has what it needs"
$SSH "$HOST" 'cd ~/qwen-0.8b/wsp \
  && [ -d eval/heldout2 ] \
  && grep -q WSP_EVAL_DIR eval_heldout.py \
  && ls ~/qwen-0.8b/*.safetensors >/dev/null 2>&1 \
  && echo VERIFIED' < /dev/null | grep -q VERIFIED \
  || die "the pod is missing the replacement set, the patched harness, or the 0.8B weights. Nothing launched."
say "Verified."

say "4/4  Launching, detached from this terminal and from the Mac"
$SSH "$HOST" "cd ~/qwen-0.8b/wsp && setsid nohup bash gpu/remeasure_v2.sh > ~/qwen-0.8b/remeasure_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 12
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/REMEASURE_STATUS 2>/dev/null; echo "---"; tail -30 ~/qwen-0.8b/remeasure_v2_*.log 2>/dev/null | tail -30' < /dev/null

cat <<'TAIL'

=========================================================
Re-measurement running on the pod. You can close the Mac.
Expect 1-2 hours. No API spend.

Check on it:
  ssh -i ~/qwen-0.8b/.ssh_runpod/id_ed25519 -p 11873 root@POD_IP \
    'cat ~/qwen-0.8b/REMEASURE_STATUS; tail -60 ~/qwen-0.8b/remeasure_v2_*.log'

When it says FINISHED:
  double-click  ~/qwen-0.8b/wsp/gpu/pull_remeasure.command

IMPORTANT: it runs a grader validation FIRST and prints a digest of
failures. Those numbers mean nothing until Claude has looked at that
digest - a wrong grader fails in a way that looks perfectly fine.

Your original eval/heldout/ is NOT overwritten; the replacement set is
staged separately, so the old numbers stay comparable.
=========================================================
TAIL
echo "Press return to close."; read _
