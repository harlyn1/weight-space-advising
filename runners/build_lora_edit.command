#!/bin/bash
# DOUBLE-CLICK THIS ON THE MAC. Converts the merged LoRA model into the harness's own
# per-tensor format, so the fine-tune arm can be measured like every weight-edit arm.
#
# Runs entirely on the Mac. Touches no pod, spends nothing, starts nothing.
# Safe to run while something else is going on the pod.
#
#   finetune/merged/ (1.5 GB)  ->  edited/*.npy  ->  runs/full_finetune/edited_final/
#
# Once this has run, the fine-tune arm travels like any other frozen edit: only the
# changed tensors cross the wire, never the 1.5 GB.
SRC=~/qwen-0.8b/wsp
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }
die () { printf '\n\033[31mSTOPPED: %s\033[0m\n' "$*"; echo; echo "Press return to close."; read _; exit 1; }

cd "$SRC" || die "no $SRC on this Mac"

say "1/4  Checking what is here"
[ -d finetune/merged ]          || die "finetune/merged is not on this Mac. Nothing to convert."
[ -f finetune/merged_to_edits.py ] || die "finetune/merged_to_edits.py is missing."
PY="$SRC/.venv/bin/python"
[ -x "$PY" ] || PY=python3
"$PY" -c "import numpy, safetensors" 2>/dev/null || die "$PY cannot import numpy and safetensors. Try: $SRC/.venv/bin/python -m pip install numpy safetensors"
echo "     merged model: $(du -sh finetune/merged | cut -f1)"
echo "     interpreter:  $PY"

if ls runs/full_finetune/edited_final/*.npy >/dev/null 2>&1; then
  say "Already built: $(ls runs/full_finetune/edited_final/*.npy | wc -l | tr -d ' ') tensors in runs/full_finetune/edited_final"
  echo "Delete that directory first if you want it rebuilt."
  echo; echo "Press return to close."; read _; exit 0
fi

say "2/4  Clearing the shared staging directory"
# edited/ is scratch shared by every arm, and it currently holds another arm's tensors.
# merged_to_edits.py copies only the names it writes, so the snapshot would be correct
# either way - but stale edited/ state has caused real bugs in this project.
N=$(ls edited/*.npy 2>/dev/null | wc -l | tr -d ' ')
if [ "${N:-0}" -gt 0 ]; then
  echo "     moving $N leftover tensor(s) aside rather than deleting them:"
  mkdir -p edited_previous && mv edited/*.npy edited_previous/ 2>/dev/null
  ls edited_previous/*.npy 2>/dev/null | sed 's|.*/|       |'
else
  mkdir -p edited; echo "     already empty"
fi

say "3/4  Converting - it compares every tensor against the original, a few minutes"
"$PY" finetune/merged_to_edits.py || die "merged_to_edits.py failed. The output above says why; nothing else was changed."

say "4/4  Result"
M=$(ls runs/full_finetune/edited_final/*.npy 2>/dev/null | wc -l | tr -d ' ')
[ "${M:-0}" -gt 0 ] || die "the script finished but wrote no tensors to runs/full_finetune/edited_final"
echo "     $M tensors in runs/full_finetune/edited_final"
[ -f runs/full_finetune/edit_summary.json ] && "$PY" -c "
import json; d=json.load(open('runs/full_finetune/edit_summary.json'))
print('     bf16 entries changed:', format(d.get('bf16_entries_changed',0), ','))
print('     source:', d.get('source'))" 2>/dev/null
rm -f edited/*.npy 2>/dev/null

cat <<'TAIL'

=========================================================
The fine-tune arm is now in the same format as every edit.

It gets measured on the clean set the next time the combos
run - gpu/remeasure_combos.sh picks it up automatically.

Your previous staging tensors, if there were any, are in
  ~/qwen-0.8b/wsp/edited_previous/
and can be deleted once you are satisfied.
=========================================================
TAIL
echo "Press return to close."; read _
