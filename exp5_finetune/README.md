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
| re-weighted | 200 | 40.0% | 65.5 | 92.9 | **7.0%** | (on pod) |
| rejection-sampled | 200 | 40.0% | — | — | — | (running) |

**The original failed on aim, not budget.** It cost $2.71, dev loss plateaued by epoch 2
(1.760 → 1.738), and only 16 of its 225 examples taught the model to defer. Its median answer
was 1,433 characters and 9% opened with "I" — so it learned the shape of a thorough advising
answer, which is why it is the one lever that *raised* dangerous answers.

**Re-weighted to 40% fabrication, it becomes the best arm in the experiment**, matching Sonnet's
edit on fabrication exactly (+53.6 both) and beating it on dangerous answers, 7.0% against 9.1%.
It also beats the edit significantly on GPA reading (+21.9 ±9.8 vs +6.2 ±10.0) and on demographic
steering (+11.1 ±5.1 vs &minus;4.2 ±6.4), while losing on general facts. GPA and steering were
*downsampled* in that curriculum and still improved, which points at interference in the original
mix rather than scarcity.

The caveat, and why the third run exists: those 80 fabrication slots are **16 unique examples
repeated five times** — the entire practice fabrication pool. `code/rejection_sample.py` replaces
them with fresh, short refusals the unedited model produced itself, kept only where the eval's own
grader marks them correct, to separate learning a behaviour from memorising sixteen answers.

What this costs the headline: a competently aimed fine-tune beats the best weight edit on this
task. What survives is that the edit matched it with **18,432 weights against 257 million** and
no training data at all.

> **Pending fetch.** Some eval files listed below were produced on the GPU pod and have not
> been copied into this repo yet. They are marked `(on pod)`. The pod's disk is ephemeral;
> `runners/pull_results.command` retrieves them.
