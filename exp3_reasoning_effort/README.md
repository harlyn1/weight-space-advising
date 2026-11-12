<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp3 — pass budget against reasoning effort

The paper reported that its fix needed 50 passes and barely moved at 10 or 25. This tests the
other end: what more budget buys, and what more thinking per turn buys instead.

Clean set = 141 replacement items written after the leak was found; no editor ever saw them. Base scores 50.7 / 39.3 / 28.9%.
Original set = the compromised held-out set, where four editors could read graded answers. Base scores 62.5 / 57.1 / 15.3%. **Never compare a number across the two.**

| run | passes | effort | mean | fabrication | severe | build | eval file |
|---|---|---|---|---|---|---|---|
| Astra | 40 | high | 54.3 | 53.6 | 23.2% | $13.70 | `../exp2_editors/eval/astra.clean.json` |
| Astra resumed | 80 | high | 54.3 | 53.6 | 23.2% | $23.87 | `eval/astra80.clean.json` |
| Astra | 40 | xhigh | 53.7 | 64.3 | 20.4% | $12.22 | `eval/astra_xhigh.clean.json` |

**Doubling the budget produced a byte-identical file** — same md5 on all three tensors — after
spending the extra 40 passes building and rejecting four more constructions. $10.17 for nothing.

Raising reasoning effort one notch on the same 40 passes moved fabrication +10.7 and dangerous
answers &minus;2.8, for six tenths off the mean, and cost *less* because it took fewer turns.

`eval/astra_published.original.json` is the paper's own published loop fix, measured here.
