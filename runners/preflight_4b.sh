#!/bin/bash
# The 4B pre-flight. Run ON THE BOX after gpu/setup_4b.sh:
#   cd ~/qwen-4b/wsp && nohup bash gpu/preflight_4b.sh > work/preflight.log 2>&1 &
# Answers, before any editor spend: does this model still fail the way the 0.8B did, which failures are
# editable, is there an honesty-opener neuron, and which from-scratch constructions have a readable trigger.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p work
step () { echo; echo "===== $* ====="; date -u +%H:%M:%S; }

step "0. architecture"
$PY -c "
import sys; sys.path.insert(0,'.')
import kernel as K, server as S
print(K.ARCH)
print('derived:', S.N_LAYERS, 'layers, d_model', S.D_MODEL, 'd_ff', S.D_FF, 'full-attn', S.FULL_ATTN)"

step "1. base diagnostic on the practice split (what it gets wrong, and how)"
$PY eval_heldout.py --arm pre4b_practice --weights original --n 1 --no-tf \
  --sets facts scenarios gpa_values eligibility fabrication calibration steering || echo "(practice diagnostic failed)"

step "2. base row on the evaluation sets (the 4B's starting point)"
$PY eval_heldout.py --arm pre4b_base --weights original --n 2 || echo "(base row failed)"

step "3. mechanism vs knowledge: the frame-prepend classifier"
$PY gpu/frame_test.py --n 2 || echo "(frame test failed)"

step "4. the honesty-opener neuron search, last three layers"
$PY gpu/opener_probe.py --layers 3 --n 24 || echo "(opener probe failed)"

step "5. decodability of the from-scratch triggers"
$PY gpu/decodability.py --every 4 || echo "(decodability failed)"

step "done"
echo "reports: work/eval_pre4b_base/summary.md  work/frame_test/report.json  work/opener_probe/report.json  work/decodability/report.json"
