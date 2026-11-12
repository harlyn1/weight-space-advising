#!/bin/bash
# Run on the Mac: copy the weights and the harness (with your .env) to a rented box over ssh.
#   bash ~/qwen-0.8b/wsp/gpu/push.sh user@host [port]        e.g.  bash gpu/push.sh ubuntu@1.2.3.4 22
set -e
HOST=${1:?usage: push.sh user@host [port]}; PORT=${2:-22}
KEY=${WSP_SSH_KEY:-~/qwen-0.8b/.ssh_runpod/id_ed25519}
SRC="$(cd "$(dirname "$0")/../.." && pwd)"          # ~/qwen-0.8b
echo "copying $SRC -> $HOST:~/qwen-0.8b (weights ~1.6 GB + harness; .venv and merged models excluded)"
rsync -az --progress -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude '.venv' --exclude 'wsp/finetune/merged*' --exclude 'wsp/__pycache__' --exclude 'wsp/*/__pycache__' \
  --exclude 'wsp/work/*_acts*' --exclude 'wsp/requests/*.claimed' \
  "$SRC/" "$HOST:~/qwen-0.8b/"
echo "done. Next, on the box:   ssh -p $PORT $HOST   then   cd ~/qwen-0.8b/wsp && bash gpu/setup_gpu.sh"
