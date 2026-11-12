#!/bin/bash
# Run ON THE BOX once: python env, the 4B checkpoint, the size-agnostic patch, and a smoke test.
#   cd ~/qwen-4b/wsp && bash gpu/setup_4b.sh
set -e
cd "$(dirname "$0")/.."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU visible"; exit 1; }
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip wheel
pip install -q "torch>=2.6" "transformers==5.22.1" accelerate safetensors tokenizers numpy scipy pandas huggingface_hub anthropic openai
# fused Gated DeltaNet kernels: without them the 24 linear-attention layers fall back to the reference
# PyTorch path, which is what made the Mac's teacher-forced reads take hours. Report, do not fail.
pip install -q flash-linear-attention && echo "flash-linear-attention: installed" || echo "flash-linear-attention: NOT installed (slow fallback)"
pip install -q causal-conv1d      && echo "causal-conv1d: installed"           || echo "causal-conv1d: NOT installed (slow fallback)"
python - <<'PYEOF'
import torch, transformers
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
try:
    import fla; print("fla import OK")
except Exception as e:
    print("fla unavailable:", type(e).__name__)
PYEOF
# the checkpoint, beside the harness so kernel.MODEL_DIR = ~/qwen-4b
if [ ! -f ../model.safetensors ] && [ -z "$(ls ../*.safetensors 2>/dev/null)" ]; then
  echo "downloading Qwen/Qwen3.5-4B (~8 GB) into $(cd .. && pwd)"
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B', local_dir='..', allow_patterns=['*.safetensors','*.json','*.txt','tokenizer*'])"
else
  echo "checkpoint already present: $(ls ../*.safetensors | wc -l) safetensors file(s)"
fi
python gpu/patch_4b.py
echo
echo "smoke test: one short generation on the original weights"
WSP_CODE_SECONDS=600 python - <<'PYEOF'
import os, sys, time, argparse
sys.path.insert(0, os.getcwd())
import server
t0 = time.time()
S = server.Server(argparse.Namespace(budget=0, device=None, dtype="bf16", batch=8))
print(f"model loaded in {time.time()-t0:.0f} s on {S.device}")
t0 = time.time()
S.run_generate({"type": "generate", "model": "original",
                "prompts": ["What does FAFSA stand for, and what is it used for?"],
                "n": 1, "max_tokens": 128, "thinking": False}, "smoke")
print(f"generation OK in {time.time()-t0:.0f} s -> work/gen_smoke.jsonl")
PYEOF
echo
echo "OK. Next: bash gpu/preflight_4b.sh"
