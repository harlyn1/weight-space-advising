#!/bin/bash
# DOUBLE-CLICK ON THE MAC when the clean Fable re-run reports FINISHED.
# Fetches it into ~/qwen-0.8b/wsp/gpu/box_sync/fable_clean/ so nothing local is overwritten.
HOST=root@69.8.146.173
PORT=11873
KEY=~/qwen-0.8b/.ssh_runpod/id_ed25519
DEST=~/qwen-0.8b/wsp/gpu/box_sync/fable_clean
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
RS="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
say () { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Status on the pod:"
$SSH "$HOST" 'cat ~/qwen-0.8b/FABLE_CLEAN_STATUS 2>/dev/null || echo "(no status file - was it launched?)"' < /dev/null

mkdir -p "$DEST"
say "Fetching (results, transcript, edit - no model weights)"
for d in work/eval_fable_clean runs/full_fable_clean runs/full_fable_clean.log; do
  rsync -az -e "$RS" --exclude '*.safetensors' --exclude '*.bin' \
    "$HOST:~/qwen-0.8b/wsp/$d" "$DEST/" 2>/dev/null && echo "  fetched $d" || echo "  (no $d yet)"
done
rsync -az -e "$RS" "$HOST:~/qwen-0.8b/fable_clean_*.log" "$DEST/" 2>/dev/null && echo "  fetched run log" || true

say "LEAK AUDIT from the run log:"
sed -n '/LEAK AUDIT/,/VERDICT/p' "$DEST"/fable_clean_*.log 2>/dev/null | sed 's/^/  /' | tail -20

say "Clean numbers vs the original contaminated row:"
python3 - "$DEST/eval_fable_clean/summary.json" <<'PY' 2>/dev/null || echo "  (no summary yet)"
import json, sys
s = json.load(open(sys.argv[1])); st = s.get('sets', {})
orig = {'heldout':56.7,'heldout_gpa':62.5,'heldout_eligibility':52.3,
        'heldout_fabrication':89.3,'heldout_calibration':71.4,'heldout_steering':75.0}
print("  %-22s %8s %8s %8s" % ("family", "clean", "orig", "delta"))
for k, v in st.items():
    c = v.get('acc', 0) * 100
    o = orig.get(k)
    d = ("%+.1f" % (c - o)) if o is not None else "-"
    print("  %-22s %8.1f %8s %8s" % (k.replace('heldout_', '').replace('heldout', 'facts'), c, o if o is not None else "-", d))
print("  %-22s %8.1f %8.1f %+8.1f" % ("MEAN", s.get('mean_acc', 0) * 100, 67.9, s.get('mean_acc', 0) * 100 - 67.9))
print("  %-22s %8.1f %8.1f %+8.1f" % ("severe-error rate", s.get('severe_rate', 0) * 100, 10.9, s.get('severe_rate', 0) * 100 - 10.9))
PY

cat <<'TAIL'

=========================================================
Pulled into ~/qwen-0.8b/wsp/gpu/box_sync/fable_clean/
Tell Claude it's down and it will fold this into the comparison page.

A large negative delta on fabrication means the original +32.1
was substantially leak-inflated. A small one means the edit holds.

Once pulled and happy, you can stop the pod to stop the billing.
=========================================================
TAIL
echo "Press return to close."; read _
