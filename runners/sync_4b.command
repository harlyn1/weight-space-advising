#!/bin/bash
# Pulls the box's work/ folder here every 2 minutes so Claude can read the pre-flight's progress.
# Safe to run alongside anything else; it only reads from the box. Close the window to stop it.
HOST="root@69.8.146.173"
PORT="11873"
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"
SYNC=~/qwen-0.8b/wsp/gpu/box_sync
mkdir -p "$SYNC/work"
echo "=== $(date) syncing $HOST:~/qwen-4b/wsp/work/ -> $SYNC/work/ every 2 min"
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
