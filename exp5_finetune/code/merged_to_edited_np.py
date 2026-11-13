"""merged HF dir -> wsp/edited/*.npy, numpy only.

Same job as merged_to_edits.py but without torch and without the second 1.2 GB
snapshot copy -- the pod ran out of disk otherwise. Compares stored bf16 bits
directly and skips frozen tensors, so embed_tokens is never materialised as f32.
"""
import glob, json, os, struct, sys
import numpy as np
sys.path.insert(0, os.path.abspath('.'))
import kernel as K
MERGED = sys.argv[1]
_DT = {'BF16': np.uint16, 'F16': np.float16, 'F32': np.float32}
def hdr(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n
names, FROZEN = set(K.NAMES), set(K.FROZEN)
changed = entries = skipped = 0
for path in sorted(glob.glob(os.path.join(MERGED, '*.safetensors'))):
    head, off = hdr(path)
    mm = np.memmap(path, dtype=np.uint8, mode='r')
    for key, meta in head.items():
        if key == '__metadata__': continue
        cands = [key, key.replace('model.layers.', 'model.language_model.layers.', 1),
                 key if key.startswith('model.language_model.')
                 else key.replace('model.', 'model.language_model.', 1)]
        nm = next((c for c in cands if c in names), None)
        if nm is None or nm in FROZEN or tuple(meta['shape']) != K.SHAPES.get(nm):
            skipped += 1; continue
        dt = meta['dtype']
        if dt not in _DT: skipped += 1; continue
        a, b = meta['data_offsets']
        new_raw = np.frombuffer(mm[off + a: off + b], dtype=_DT[dt]).reshape(tuple(meta['shape']))
        old_raw, old_dt = K._raw(nm)
        if dt == 'BF16' and old_dt == 'BF16':
            d = int((new_raw != old_raw).sum())
            if d == 0: continue
            new32 = K.bf16_to_f32(new_raw)
        else:
            new32 = K.bf16_to_f32(new_raw) if dt == 'BF16' else new_raw.astype(np.float32)
            d = int((K.bf16_bits(new32) != K.bf16_bits(K.load_original(nm))).sum())
            if d == 0: continue
        try: K.save_edit(nm, new32)
        except Exception: skipped += 1; del new32; continue
        changed += 1; entries += d; del new32
    del mm
print('tensors changed:', changed, '| bf16 entries:', format(entries, ','), '| skipped:', skipped)
