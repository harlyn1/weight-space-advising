<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp2 — five editors, 40 forward passes each

Each editor received the same brief (`prompt_full.md`, `prompt_improve_advising.md`), the
paper's loop fix as a starting point, and read/write access to the weights. None could see
another's work. Each arm's own written conclusion is in `<arm>/final_answer.md`.

Clean set = 141 replacement items written after the leak was found; no editor ever saw them. Base scores 50.7 / 39.3 / 28.9%.
Original set = the compromised held-out set, where four editors could read graded answers. Base scores 62.5 / 57.1 / 15.3%. **Never compare a number across the two.**

| editor | weights changed | mean | fabrication | severe | build | eval file |
|---|---|---|---|---|---|---|
| Sonnet 5.5 | 18,432 | 60.9 | 92.9 | 9.1% | $87.78 | `eval/sonnet.clean.json` |
| Opus 5.5 | 6,144 | 56.4 | 89.3 | 18.9% | $106.39 | `eval/opus.clean.json` |
| Fable 5.1 | 60,000,000 | 55.1 | 71.4 | 23.5% | $110.73 | `eval/fable.clean.json` |
| GPT-6 Astra | 3,072 | 54.3 | 53.6 | 23.2% | $13.70 | `eval/astra.clean.json` |
| Kimi K3 | 0 (declined) | 50.9 | 39.3 | 30.4% | $18.48 | `eval/kimi.clean.json` |

Four of the five independently located the same component: one neuron in the last MLP whose
gate reads "asks for a specific the model cannot know" with the wrong sign (exp8).

**Provenance matters here.** Fable and Opus both read the held-out files *before* writing a
single weight, and every save came after. Sonnet had zero exposure across 383 turns. See exp7.
Astra's arms are in exp3.
