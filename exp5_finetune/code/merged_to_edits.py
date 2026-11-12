"""Arm D, step 3: turn the merged fine-tuned copy into weight edits (the harness's own format), so it is evaluated exactly
like the weight-editing arms: the original model loaded once, the changed tensors applied through edited/.

    cd ~/qwen-0.8b/wsp && .venv/bin/python finetune/merged_to_edits.py            # finetune/merged -> edited/*.npy -> runs/full_finetune/edited_final/

Reads finetune/merged/*.safetensors, compares every tensor with the original weights, and saves the ones that changed
(the 96 LoRA-touched projections) with kernel.save_edit; prints the count and the total number of changed bf16 entries.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

WSP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, WSP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=os.path.join(WSP, "finetune", "merged"))
    ap.add_argument("--snapshot", default=os.path.join(WSP, "runs", "full_finetune", "edited_final"))
    args = ap.parse_args()
    import kernel as K
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(args.merged, "*.safetensors")))
    assert files, f"no safetensors in {args.merged}"
    names = set(K.NAMES)
    changed, entries, skipped = [], 0, []
    for f in files:
        with safe_open(f, "pt") as st:
            for key in st.keys():
                # the merged copy was saved from the text-only class: model.layers.N... ; the originals are model.language_model.layers.N...
                cands = [key, key.replace("model.layers.", "model.language_model.layers.", 1),
                         key.replace("model.", "model.language_model.", 1) if not key.startswith("model.language_model.") else key]
                name = next((c for c in cands if c in names), None)
                if name is None:
                    skipped.append(key)
                    continue
                new = st.get_tensor(key).float().numpy()
                old = K.load_original(name)
                if new.shape != old.shape:
                    skipped.append(f"{key} (shape {new.shape} vs {old.shape})")
                    continue
                if np.array_equal(K.bf16_bits(new), K.bf16_bits(old)):
                    continue
                K.save_edit(name, new)
                changed.append(name)
                entries += int((K.bf16_bits(new) != K.bf16_bits(old)).sum())
    print(f"{len(changed)} tensors changed, {entries:,} bf16 entries; {len(skipped)} tensors not matched/skipped")
    if skipped[:5]:
        print("  e.g.", skipped[:5])
    assert changed, "no tensor differs from the original: nothing to evaluate"
    os.makedirs(args.snapshot, exist_ok=True)
    for name in changed:
        shutil.copy(os.path.join(WSP, "edited", name + ".npy"), args.snapshot)
    json.dump({"tensors": changed, "bf16_entries_changed": entries, "source": args.merged},
              open(os.path.join(os.path.dirname(args.snapshot), "edit_summary.json"), "w"), indent=1)
    print(f"snapshot: {args.snapshot}")


if __name__ == "__main__":
    main()
