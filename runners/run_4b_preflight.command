#!/bin/bash
# Double-click this (or run it) on the Mac. It pushes the harness to the rented box, sets the box up,
# starts the 4B pre-flight there, and then syncs results back here every 3 minutes so Claude can read them.
# Everything after the first minute is unattended; you can close this window once it says "pre-flight started".
HOST="root@69.8.146.173"
PORT="11873"
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
SYNC=~/qwen-0.8b/wsp/gpu/box_sync
cd ~/qwen-0.8b/wsp || exit 1
mkdir -p "$SYNC"
exec > >(tee -a "$SYNC/launch.log") 2>&1
echo "=== $(date) launching the 4B pre-flight on $HOST:$PORT"

# If the pre-flight is already running on the box (a previous launch got as far as starting it), do not
# push or set up again - just resume syncing its results back. Makes this file safe to re-open any time.
if $SSH "$HOST" "test -f ~/qwen-4b/wsp/work/preflight.log" 2>/dev/null; then
  echo "pre-flight already running on the box; skipping push/setup and resuming the sync"
  mkdir -p "$SYNC/work"
  while true; do
    rsync -az -e "$SSH" --exclude '*.safetensors' --exclude '.venv' \
      "$HOST:~/qwen-4b/wsp/work/" "$SYNC/work/" 2>> "$SYNC/sync_errors.log" \
      && echo "$(date +%H:%M:%S) synced ($(ls "$SYNC/work" | wc -l | tr -d ' ') entries)" \
      || echo "$(date +%H:%M:%S) sync failed, see sync_errors.log"
    if [ -f "$SYNC/work/preflight.log" ] && grep -q "===== done =====" "$SYNC/work/preflight.log"; then
      echo "=== $(date) pre-flight finished; final sync done"; break
    fi
    sleep 120
  done
  exit 0
fi

echo "--- 1/4 pushing the harness (no weights, no .env)"
bash gpu/push_4b.sh "$HOST" "$PORT" || { echo "PUSH FAILED"; exit 1; }

echo "--- 2/4 setting the box up (python env, 4B checkpoint, size-agnostic patch, smoke test)"
$SSH "$HOST" "cd ~/qwen-4b/wsp && bash gpu/setup_4b.sh" || { echo "SETUP FAILED"; exit 1; }

echo "--- 3/4 starting the pre-flight detached on the box"
$SSH "$HOST" "cd ~/qwen-4b/wsp && mkdir -p work && nohup bash gpu/preflight_4b.sh > work/preflight.log 2>&1 < /dev/null & disown; echo started"
echo "pre-flight started. You can close this window; the sync below keeps running until the job finishes."

echo "--- 4/4 syncing results back to $SYNC every 3 minutes"
while true; do
  rsync -az -e "$SSH" --exclude '*.safetensors' --exclude '.venv' \
    "$HOST:~/qwen-4b/wsp/work/" "$SYNC/work/" 2>/dev/null
  if [ -f "$SYNC/work/preflight.log" ] && grep -q "===== done =====" "$SYNC/work/preflight.log"; then
    echo "=== $(date) pre-flight finished; final sync done"; break
  fi
  sleep 180
done
