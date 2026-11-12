#!/bin/bash
# Pushes tonight's fixes to the box and re-runs the two things that were wrong:
#   1. server.py  - the evaluation can resolve held-out ids again (they live outside practice/ now)
#   2. the base row on the evaluation sets, which failed on that
#   3. opener_probe.py - anchored on the real generation prompt, aborts if the anchor is still wrong
# The decodability probes stay skipped: their sample sizes are too small to mean anything.
HOST="root@69.8.146.173"
PORT="11873"
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"
SYNC=~/qwen-0.8b/wsp/gpu/box_sync
cd ~/qwen-0.8b/wsp || exit 1
mkdir -p "$SYNC/work"
exec > >(tee -a "$SYNC/fix.log") 2>&1
echo "=== $(date) pushing fixes and re-running"

for f in server.py gpu/opener_probe.py gpu/decodability.py envfile.py; do
  scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new "$f" "$HOST:~/qwen-4b/wsp/$f" || { echo "SCP FAILED on $f"; exit 1; }
  echo "pushed $f"
done

echo "--- re-applying the size-agnostic patch to the fresh server.py, then re-running"
$SSH "$HOST" "cd ~/qwen-4b/wsp && nohup bash -c '
  .venv/bin/python gpu/patch_4b.py
  echo \"===== 2c. base row on the evaluation sets =====\"; date -u +%H:%M:%S
  .venv/bin/python eval_heldout.py --arm pre4b_base --weights original --n 2
  echo \"===== 4c. opener neuron probe =====\"; date -u +%H:%M:%S
  .venv/bin/python gpu/opener_probe.py --layers 3 --n 24
  echo \"===== 5c. decodability =====\"; date -u +%H:%M:%S
  .venv/bin/python gpu/decodability.py --every 4
  echo \"===== done3 =====\"; date -u +%H:%M:%S
' > work/preflight3.log 2>&1 < /dev/null & disown; echo started"

echo "--- syncing every 2 min until it finishes"
while true; do
  rsync -az -e "$SSH" --exclude '*.safetensors' --exclude '.venv' "$HOST:~/qwen-4b/wsp/work/" "$SYNC/work/" 2>/dev/null \
    && echo "$(date +%H:%M:%S) synced ($(ls "$SYNC/work" | wc -l | tr -d ' ') entries)"
  if grep -q "===== done3 =====" "$SYNC/work/preflight3.log" 2>/dev/null; then echo "=== $(date) finished"; break; fi
  sleep 120
done
