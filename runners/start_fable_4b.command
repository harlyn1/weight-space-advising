#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Starts the Fable 4B editor arm on the rented pod.
# Once it reports "launched", the run lives on the POD: you can close this window,
# close the Mac, and lose internet. Nothing on the Mac is needed again until you pull results.
#
# THIS IS THE WEIGHT-SPACE-PROGRAMMING 4B POD. It is NOT the insurance RSI pod.
HOST=root@POD_IP
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
CAP=${CAP:-200}            # hard spend cap in USD, enforced inside editor_api.py
BUDGET=${BUDGET:-40}       # forward passes

SRC=~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

say "Target: $HOST port $PORT  (WSP 4B pod - not the insurance pod)"
say "Spend cap \$$CAP · $BUDGET passes · editor Fable 5.1"

$SSH "$HOST" 'echo ok' >/dev/null 2>&1 || die "cannot reach the pod. Is it running in the RunPod console?"
say "Pod reachable."

say "1/4  Copying the pieces the push script leaves out (.env, envfile.py, the 4B brief, the runner)"
# Only ANTHROPIC_API_KEY is sent - the run uses Fable alone. The OpenAI and Fireworks keys in
# your .env stay on this Mac. The value is piped straight into the pod: never printed, never
# written to a temp file, never copied as a whole file.
grep '^ANTHROPIC_API_KEY=' "$SRC/.env" \
  | $SSH "$HOST" 'umask 077; cat > ~/qwen-4b/wsp/.env' \
  || die "could not send ANTHROPIC_API_KEY"
$SSH "$HOST" 'grep -c "^ANTHROPIC_API_KEY=.\{20,\}" ~/qwen-4b/wsp/.env' >/dev/null \
  || die "the key did not land on the pod"
scp -q -i "$KEY" -P "$PORT" "$SRC/envfile.py"            "$HOST:~/qwen-4b/wsp/envfile.py"  || die "could not copy envfile.py"
scp -q -i "$KEY" -P "$PORT" "$SRC/tasks/q2_full_4b.md"   "$HOST:~/qwen-4b/wsp/tasks/"      || die "could not copy the brief"
scp -q -i "$KEY" -P "$PORT" "$SRC/gpu/fable_4b.sh"       "$HOST:~/qwen-4b/wsp/gpu/"        || die "could not copy the runner"
scp -q -i "$KEY" -P "$PORT" "$SRC/server.py"             "$HOST:~/qwen-4b/wsp/server.py"   || die "could not copy server.py"
scp -q -i "$KEY" -P "$PORT" "$SRC/eval_heldout.py"       "$HOST:~/qwen-4b/wsp/eval_heldout.py" || die "could not copy eval_heldout.py"
$SSH "$HOST" 'cd ~/qwen-4b/wsp && python3 gpu/patch_4b.py' 2>&1 | tail -3
$SSH "$HOST" 'chmod 600 ~/qwen-4b/wsp/.env; chmod +x ~/qwen-4b/wsp/gpu/fable_4b.sh'
say "Copied. Pod .env holds ONLY ANTHROPIC_API_KEY, mode 600."

say "2/4  Checking the leak guard on the pod"
if ! $SSH "$HOST" 'grep -q "held-out ids are not available to the editor" ~/qwen-4b/wsp/server.py' 2>/dev/null; then
  say "   Guard ABSENT on the pod. Re-pushing server.py and eval_heldout.py from this Mac."
  scp -q -i "$KEY" -P "$PORT" "$SRC/server.py"       "$HOST:~/qwen-4b/wsp/server.py"       || die "could not re-push server.py"
  scp -q -i "$KEY" -P "$PORT" "$SRC/eval_heldout.py" "$HOST:~/qwen-4b/wsp/eval_heldout.py" || die "could not re-push eval_heldout.py"
  $SSH "$HOST" 'cd ~/qwen-4b/wsp && python3 gpu/patch_4b.py' 2>&1 | tail -5
  $SSH "$HOST" 'grep -q "held-out ids are not available to the editor" ~/qwen-4b/wsp/server.py' \
    || die "the guard is still missing after re-pushing. NOT launching - this is the fault that invalidated the first experiment."
  say "   Guard restored and the harness re-patched for 4B."
else
  say "   Guard present."
fi

say "3/4  Clearing any leftover server or editor on the pod"
$SSH "$HOST" 'pkill -f "server.py --budget" 2>/dev/null; pkill -f "editor_api.py --task" 2>/dev/null; sleep 2; true'

say "4/4  Launching, detached from this terminal and from the Mac"
$SSH "$HOST" "cd ~/qwen-4b/wsp && BUDGET=$BUDGET WSP_MAX_COST_USD=$CAP setsid nohup bash gpu/fable_4b.sh > ~/qwen-4b/fable_4b_launch.log 2>&1 < /dev/null & disown; sleep 3; echo launched" < /dev/null
sleep 8
say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-4b/FABLE_4B_STATUS 2>/dev/null; echo "---"; tail -25 ~/qwen-4b/fable_4b_*.log 2>/dev/null | tail -25' < /dev/null

cat <<'TAIL'

=========================================================
The run is now on the pod. You can close the Mac.

To check on it later (paste in Terminal):
  ssh -i ~/qwen-0.8b/.ssh_runpod/id_ed25519 -p 11873 root@POD_IP \
    'cat ~/qwen-4b/FABLE_4B_STATUS; tail -40 ~/qwen-4b/fable_4b_*.log'

To pull results when it says FINISHED:
  double-click  ~/qwen-0.8b/wsp/gpu/pull_4b.command

DO NOT stop the pod in the RunPod console: its disk is ephemeral
and stopping it erases the run.
=========================================================
TAIL
echo "Press return to close."; read _
