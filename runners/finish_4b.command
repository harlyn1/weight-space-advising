#!/bin/bash
# Catch-up for the two pre-flight steps that failed because envfile.py was excluded from the push
# (it is a key LOADER, no secrets; .env is still never copied). Waits for the running pre-flight to
# finish so the two jobs never share the GPU, then runs the practice diagnostic and the base row,
# and keeps syncing results back. Safe to re-open: it skips work that is already done.
HOST="root@69.8.146.173"
PORT="11873"
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20"
SYNC=~/qwen-0.8b/wsp/gpu/box_sync
cd ~/qwen-0.8b/wsp || exit 1
mkdir -p "$SYNC/work"
exec > >(tee -a "$SYNC/finish.log") 2>&1
echo "=== $(date) catch-up starting"

echo "--- copying envfile.py (loader only; .env is NOT copied)"
scp -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=accept-new envfile.py "$HOST:~/qwen-4b/wsp/envfile.py" || { echo "SCP FAILED"; exit 1; }

echo "--- waiting for the running pre-flight to finish (checking every 2 min)"
while true; do
  rsync -az -e "$SSH" --exclude '*.safetensors' --exclude '.venv' "$HOST:~/qwen-4b/wsp/work/" "$SYNC/work/" 2>/dev/null
  if grep -q "===== done =====" "$SYNC/work/preflight.log" 2>/dev/null; then echo "pre-flight finished"; break; fi
  echo "$(date +%H:%M:%S) still running: $(tail -n 1 "$SYNC/work/preflight.log" 2>/dev/null | cut -c1-80)"
  sleep 120
done

echo "--- running the two missed steps detached on the box"
$SSH "$HOST" "cd ~/qwen-4b/wsp && nohup bash -c '
  echo \"===== 1b. base diagnostic on the practice split =====\"; date -u +%H:%M:%S
  .venv/bin/python eval_heldout.py --arm pre4b_practice --weights original --n 1 --no-tf --sets facts scenarios gpa_values eligibility fabrication calibration steering
  echo \"===== 2b. base row on the evaluation sets =====\"; date -u +%H:%M:%S
  .venv/bin/python eval_heldout.py --arm pre4b_base --weights original --n 2
  echo \"===== done2 =====\"; date -u +%H:%M:%S
' > work/preflight2.log 2>&1 < /dev/null & disown; echo started"

echo "--- syncing every 2 min until those finish"
while true; do
  rsync -az -e "$SSH" --exclude '*.safetensors' --exclude '.venv' "$HOST:~/qwen-4b/wsp/work/" "$SYNC/work/" 2>/dev/null \
    && echo "$(date +%H:%M:%S) synced ($(ls "$SYNC/work" | wc -l | tr -d ' ') entries)"
  if grep -q "===== done2 =====" "$SYNC/work/preflight2.log" 2>/dev/null; then echo "=== $(date) catch-up finished"; break; fi
  sleep 120
done
