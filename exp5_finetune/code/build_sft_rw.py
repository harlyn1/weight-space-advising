"""Re-weight the SFT curriculum: fabrication from 7.1% to 40%, same training code.

Upsamples the 16 hand-written fabrication examples 5x and downsamples everything
else to 120, giving 200 examples. NOTE: this is the curriculum whose result did
NOT survive its control -- see rejection_sample.py and the exp5 README.
"""
import json, random, collections
random.seed(0)
rows = [json.loads(l) for l in open('finetune/sft.jsonl')]
fab = [r for r in rows if r['source'] == 'fab']
oth = [r for r in rows if r['source'] != 'fab']
UP = 5
TARGET_OTHER = 120
fab_up = []
for i in range(UP):
    for r in fab:
        q = dict(r); q['id'] = '%s_rw%d' % (r['id'], i); fab_up.append(q)
random.shuffle(oth)
bysrc = collections.defaultdict(list)
for r in oth: bysrc[r['source']].append(r)
keep = []
for s, rs in bysrc.items():
    n = max(1, round(TARGET_OTHER * len(rs) / len(oth)))
    keep.extend(rs[:n])
keep = keep[:TARGET_OTHER]
out = fab_up + keep
random.shuffle(out)
with open('finetune/sft_rw.jsonl', 'w') as f:
    for r in out: f.write(json.dumps(r) + chr(10))
c = collections.Counter(r['source'] for r in out)
print('examples:', len(out), '| fab share: %.1f%%' % (100 * c['fab'] / len(out)))
print('unique fab items: %d, each repeated %dx' % (len(fab), UP))
