<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp5 — LoRA fine tuning, and what its curriculum decides

Three runs of the same `code/train_lora.py` at identical hyperparameters (r=16, lr 1e-4,
3 epochs, batch 1 × accum 8, 6,389,760 trainable parameters). **Only the training data differs.**

Clean set = 141 replacement items written after the leak was found; no editor ever saw them. Base scores 50.7 / 39.3 / 28.9%.
Original set = the compromised held-out set, where four editors could read graded answers. Base scores 62.5 / 57.1 / 15.3%. **Never compare a number across the two.**

| run | examples | fabrication share | mean | fabrication | severe | eval file |
|---|---|---|---|---|---|---|
| original | 225 | 7.1% | 61.8 | 60.7 | 21.9% | (on pod) |
| re-weighted | 200 | 40.0% | 65.5 | 92.9 | 7.0% | (on pod) |
| rejection-sampled | 174 | 31.0% | 54.2 | 50.0 | 23.7% | (on pod) |

**The original failed on aim, not budget.** It cost $2.71, dev loss plateaued by epoch 2
(1.760 → 1.738), and only 16 of its 225 examples taught the model to defer. Its median answer
was 1,433 characters and 9% opened with "I" — so it learned the shape of a thorough advising
answer, which is why it is the one lever that *raised* dangerous answers.

**Re-weighting to 40% fabrication appeared to make it the best arm in the experiment** —
65.5 mean, +53.6 on fabrication matching Sonnet's edit exactly, and 7.0% dangerous answers
against the edit's 9.1%. That result does not survive its own control.

**The control, and what it found.** Those 80 fabrication slots were 16 unique hand-written
examples repeated five times — the entire practice fabrication pool.
`code/rejection_sample.py` replaces them with 54 fresh refusals the *unedited model produced
itself*, on the same 16 questions, kept only where the eval's own grader marks them correct
(`grade_checks(answer, text)[0]`, verified to reproduce 28/28 known labels). Same questions,
same hyperparameters, different answers. **Fabrication collapses from +53.6 to +10.7**, and
dangerous answers go back to 23.7%.

The reason is visible in the generations. On the 28 fabrication rows:

| arm | distinct answers | median chars | share the same opening six words |
|---|---|---|---|
| base | 28 | 1218 | 1 / 28 |
| re-weighted | 28 | 1286 | **15 / 28** |
| rejection-sampled | 28 | 920 | 3 / 28 |
| Sonnet edit | 28 | 792 | 2 / 28 |

The re-weighted model emits one canonical opening on more than half the items. It did not learn
to defer; it learned a phrasing that satisfies the grader's `must` pattern, reinforced by seeing
sixteen answers five times each. Sonnet's edit reaches the identical +53.6 with varied phrasing
and the shortest answers of any arm.

Strictly, this shows the fine-tune's route is **fragile**, not fake — both arms pass the same
grader. Settling it properly means re-testing against paraphrased grader patterns. But a
curriculum reweighting that "beats the edit" is exactly the result one would publish without
checking the openings, so the check belongs in the protocol.

**What this costs the headline: nothing, once the control is run.** The edit's +53.6 holds with
diverse phrasing, 18,432 weights against 257 million, and no training data at all. The
fine-tune reaches the same number only by templating, and reaches +10.7 when the template is
removed.

> **Pending fetch.** Some eval files listed below were produced on the GPU pod and have not
> been copied into this repo yet. They are marked `(on pod)`. The pod's disk is ephemeral;
> `runners/pull_results.command` retrieves them.
