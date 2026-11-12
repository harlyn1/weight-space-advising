<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp1 — the unedited model, and the ceiling

What Qwen3.5-0.8B does before anything is changed, and what a frontier model with the same
prompt does, as an upper reference.

Clean set = 141 replacement items written after the leak was found; no editor ever saw them. Base scores 50.7 / 39.3 / 28.9%.
Original set = the compromised held-out set, where four editors could read graded answers. Base scores 62.5 / 57.1 / 15.3%. **Never compare a number across the two.**

| run | mean | fabrication | severe | file |
|---|---|---|---|---|
| base, clean set | 50.7 | 39.3 | 28.9% | `eval/base.clean.json` |
| base, original set | 62.5 | 57.1 | 15.3% | `eval/base.original.json` |
| frontier + context, original set | 92.9 | 92.9 | 14.9% | `eval/frontier.original.json` |

The replacement set is markedly harder than the original: 11.8 points of mean and nearly twice
the rate of dangerous answers. That difference is why the correction in exp7 changed conclusions
rather than merely confirming them.

`prompt_domain_read.md` is the brief used to have a model read the domain before any editing.
