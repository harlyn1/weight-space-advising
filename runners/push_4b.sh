#!/bin/bash
# Run on the MAC. Copies the harness (no weights, no .env, no runs) to a rented box as ~/qwen-4b/wsp,
# so that kernel.MODEL_DIR = ~/qwen-4b resolves to the 4B checkpoint the box downloads itself.
#   bash ~/qwen-0.8b/wsp/gpu/push_4b.sh root@1.2.3.4 12345
set -e
HOST=${1:?usage: push_4b.sh user@host [port]}; PORT=${2:-22}
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
SRC="$(cd "$(dirname "$0")/.." && pwd)"                      # ~/qwen-0.8b/wsp
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
echo "harness $SRC -> $HOST:~/qwen-4b/wsp   (no weights, no .env, no runs/work/results)"
$SSH "$HOST" "mkdir -p ~/qwen-4b/wsp"
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude '.venv' --exclude '__pycache__' --exclude '*/__pycache__' --exclude '*.pyc' \
  --exclude 'runs' --exclude 'work' --exclude 'results' --exclude 'exports' \
  --exclude 'edited' --exclude 'paper_edits/*.npy' --exclude 'finetune/merged*' \
  --exclude '_leak_quarantine' --exclude 'requests/*' --exclude '.env' --exclude 'envfile.py' \
  "$SRC/" "$HOST:~/qwen-4b/wsp/"
echo
echo "next, on the box:"
echo "  ssh -i $KEY -p $PORT $HOST"
echo "  cd ~/qwen-4b/wsp && bash gpu/setup_4b.sh"
