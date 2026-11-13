"""The control that refutes the re-weighted result.

Rejection-samples fresh refusals from the BASE model on the practice fabrication
items. Every SFT target is something the unedited model actually said, kept only
when the eval's own grader marks it correct. Nothing is synthesised and no answer
is repeated -- which is the single difference from build_sft_rw.py.

grade_checks(answer, text) returns (score, hit, viol); element 0 is the label, and
it reproduces 28/28 known row labels.
"""
import json, os, sys, torch
sys.path.insert(0, '.'); sys.argv = ['x']
import eval_heldout as E
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.expanduser('~/qwen-0.8b')
N_SAMPLES = int(os.environ.get('NS', '16'))
KEEP_PER_ITEM = int(os.environ.get('KEEP', '5'))

items = [json.loads(l) for l in open('practice/fabrication.jsonl')]
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda').eval()

kept, tried, ncorrect = [], 0, 0
for i, it in enumerate(items):
    msgs = [{'role': 'user', 'content': it['problem']}]
    # transformers 5.x: return_dict=True, or generate() raises KeyError: 'shape'
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors='pt', return_dict=True)
    enc = {k: v.to('cuda') for k, v in enc.items()}
    plen = enc['input_ids'].shape[1]
    seen, good = set(), []
    for _ in range(N_SAMPLES):
        tried += 1
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=256, do_sample=True,
                               temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
        ans = tok.decode(o[0][plen:], skip_special_tokens=True).strip()
        if not ans or ans in seen: continue
        seen.add(ans)
        # practice items carry the checks under 'answer', not 'reference'
        if E.grade_checks(it['answer'], ans)[0] == 1:
            ncorrect += 1; good.append(ans)
    good.sort(key=len)                      # prefer the SHORTEST correct refusal
    for j, a in enumerate(good[:KEEP_PER_ITEM]):
        kept.append({'id': 'rs_%s_%d' % (it.get('id', i), j), 'source': 'fab',
                     'messages': [{'role': 'user', 'content': it['problem']},
                                  {'role': 'assistant', 'content': a}]})
with open('finetune/rs_fab.jsonl', 'w') as f:
    for r in kept: f.write(json.dumps(r) + chr(10))
print('attempted:', tried, '| distinct correct:', ncorrect, '| kept:', len(kept))
