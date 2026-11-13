#!/bin/bash
# DOUBLE-CLICK ON THE MAC when the Fable 4B run reports FINISHED.
# Fetches the 4B arms from the pod into ~/qwen-0.8b/wsp/gpu/box_sync/4b/ (kept separate
# from the 0.8B results so nothing is overwritten), then prints the numbers.
HOST=root@POD_IP
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
DEST=~/qwen-0.8b/wsp/gpu/box_sync/4b
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-4b/FABLE_4B_STATUS 2>/dev/null || echo "(no status file - was it launched?)"' < /dev/null

mkdir -p "$DEST"
say "Fetching (results, transcripts, edits - no model weights)"
for d in work/eval_base work/eval_fable work/eval_fable_context \
         runs/full_fable runs/full_fable.log results; do
  rsync -az --prune-empty-dirs \
    -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
    --exclude '*.safetensors' --exclude '*.bin' \
    "$HOST:~/qwen-4b/wsp/$d" "$DEST/" 2>/dev/null && echo "  fetched $d" || echo "  (no $d yet)"
done
rsync -az -e "ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new" \
  "$HOST:~/qwen-4b/fable_4b_*.log" "$DEST/" 2>/dev/null && echo "  fetched run log" || true

say "Summaries now local:"
for f in "$DEST"/eval_*/summary.json; do
  [ -f "$f" ] || continue
  echo "--- $(basename "$(dirname "$f")") ---"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); m=d.get('by_source',d); print(json.dumps({k:(round(v['accuracy']*100,1) if isinstance(v,dict) and 'accuracy' in v else v) for k,v in list(m.items())[:12]}, indent=1)[:900])" "$f" 2>/dev/null || head -c 400 "$f"
done

say "Final answer from the editor (if it finished):"
sed -n '1,40p' "$DEST/full_fable/final_answer.md" 2>/dev/null || echo "(not written yet)"

cat <<'TAIL'

=========================================================
Pulled into ~/qwen-0.8b/wsp/gpu/box_sync/4b/
Tell Claude it's down and it will fold these into the comparison page.

Once you have pulled and are happy, THEN you can stop the pod
in the RunPod console to stop the billing.
=========================================================
TAIL
echo "Press return to close."; read _
