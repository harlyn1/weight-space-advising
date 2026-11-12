#!/bin/bash
# One-time setup on a rented Linux GPU box (Ubuntu + NVIDIA driver; any GPU with >= 16 GB). Run on the box after push.sh:
#   cd ~/qwen-0.8b/wsp && bash gpu/setup_gpu.sh
set -e
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible"; exit 1; }
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "using $PY ($($PY --version))"
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel > /dev/null
# the same library versions as the Mac (the checkpoint and the server were exercised with these)
pip install "torch>=2.6" "transformers==5.22.1" "peft>=0.21" accelerate safetensors tokenizers numpy scipy scikit-learn pandas anthropic openai
# the fused linear-attention kernels: optional, several times faster than the PyTorch fallback the Mac uses
pip install flash-linear-attention > /dev/null 2>&1 && echo "flash-linear-attention installed" || echo "flash-linear-attention not installed (fallback kernels will be used; slower but identical)"
pip install causal-conv1d > /dev/null 2>&1 && echo "causal-conv1d installed" || echo "causal-conv1d not installed (fallback)"
python - <<'PYEOF'
import torch, transformers, peft
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "| peft", peft.__version__)
PYEOF
python kernel.py
command -v tmux > /dev/null || (sudo apt-get update -qq && sudo apt-get install -y -qq tmux) || true
echo
echo "OK. Run one editor arm (with its own base arm for the paired comparison), detached:"
echo "  tmux new -d -s wsp 'cd ~/qwen-0.8b/wsp && ARMS=\"base fable\" SERVER_BATCH=32 EVAL_BATCH=32 bash run_compare.sh > results/run_compare.log 2>&1'"
echo "  tail -f ~/qwen-0.8b/wsp/results/run_compare.log"
