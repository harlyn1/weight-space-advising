"""Build the rejection-sampled curriculum: fresh model-written refusals + the same
downsampled remainder as build_sft_rw.py. Same size and composition as the
re-weighted run; only the fabrication answers differ."""
import json, random, collections
random.seed(0)
rows = [json.loads(l) for l in open('finetune/sft.jsonl')]
fab_rs = [json.loads(l) for l in open('finetune/rs_fab.jsonl')]
oth = [r for r in rows if r['source'] != 'fab']
TARGET_OTHER = 120
random.shuffle(oth)
bysrc = collections.defaultdict(list)
for r in oth: bysrc[r['source']].append(r)
keep = []
for s, rs in bysrc.items():
    n = max(1, round(TARGET_OTHER * len(rs) / len(oth)))
    keep.extend(rs[:n])
keep = keep[:TARGET_OTHER]
out = fab_rs + keep
random.shuffle(out)
with open('finetune/sft_rs.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + chr(10))
c = collections.Counter(r['source'] for r in out)
L = sorted(len(m['content']) for r in out for m in r['messages'] if m['role'] == 'assistant')
print('examples:', len(out), '| fab share: %.1f%%' % (100 * c['fab'] / len(out)))
print('median assistant chars:', L[len(L)//2])
