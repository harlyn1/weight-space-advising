<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp4 — context injection, alone and on top of each edit

`advising.md` is a 1,300-token system prompt. The model's weights are untouched in the
context-only arm; the combination arms apply a frozen edit and the prompt together.

Clean set = 141 replacement items written after the leak was found; no editor ever saw them. Base scores 50.7 / 39.3 / 28.9%.
Original set = the compromised held-out set, where four editors could read graded answers. Base scores 62.5 / 57.1 / 15.3%. **Never compare a number across the two.**

| arm | mean | fabrication | severe | eval file |
|---|---|---|---|---|
| context only | 63.4 | 82.1 | 18.9% | `eval/context.clean.json` |
| Sonnet + context | 67.9 | 92.9 | 11.9% | (on pod) |
| Opus + context | 67.9 | 96.4 | 12.5% | (on pod) |
| Kimi + context | 67.1 | 89.3 | 18.9% | (on pod) |
| Fable + context | 67.0 | 96.4 | 12.2% | (on pod) |
| Astra xhigh + context | 66.3 | 85.7 | 18.9% | (on pod) |
| Astra + context | 64.9 | 85.7 | 16.1% | (on pod) |

Two findings. **The levers are not substitutes**: the prompt tops the mean, the edit reaches a
floor the prompt cannot — adding the prompt to Sonnet's edit pushes dangerous answers from
9.1% back up to 11.9%. And **the weaker the edit, the more the prompt adds**: Kimi's inert edit
gains +16.2 from context, Sonnet's gains +7.0.

Context is also the only lever that moves federal aid eligibility (+18.8 ±8.2). No edit does.

> **Pending fetch.** Some eval files listed below were produced on the GPU pod and have not
> been copied into this repo yet. They are marked `(on pod)`. The pod's disk is ephemeral;
> `runners/pull_results.command` retrieves them.
