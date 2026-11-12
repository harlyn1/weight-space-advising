#!/bin/bash
# Run on the Mac when a box's arm has finished: fetch that arm's editor transcript, edit and evaluation into this wsp folder,
# then rebuild the table and the page.
#   bash ~/qwen-0.8b/wsp/gpu/pull.sh user@host fable [port]
set -e
HOST=${1:?usage: pull.sh user@host <arm> [port]}; ARM=${2:?arm name}; PORT=${3:-22}
WSP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WSP"
for d in "runs/full_$ARM" "work/eval_$ARM" "work/fwd_ev_${ARM}_steertf" "work/eval_${ARM}_context" "runs/full_$ARM.log"; do
  rsync -az -e "ssh -p $PORT" "$HOST:~/qwen-0.8b/wsp/$d" "$(dirname "$d")/" 2>/dev/null && echo "fetched $d" || true
done
.venv/bin/python compare_arms.py > /dev/null && .venv/bin/python make_report.py
echo "table: results/full_test.md  page: results/full_test.html"
