# Weight-Space Programming for Student Advising

Nov 11, 2026 · @H · *supersedes the Nov 11 morning memo*

> **All figures below are from a clean re-measurement.** Partway through I found four of my five editors could see the evaluation answer key through a bug in my harness. Every frozen edit was re-measured on 141 replacement items written afterwards, which no editor ever saw. Where a number exists only on the compromised set I mark it *(old set)*. Two claims in the previous memo did not survive and are withdrawn below.

## TLDR

You may have seen the [weight-space programming paper by Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/) I ran their method in a new domain, higher education student advising, with six editors instead of one. Five frontier models each got 40 forward passes to read Qwen3.5-0.8B's weights and fix what they found, across six graded families: facts and scenarios, reading a GPA or SAT, federal aid eligibility, fabrication, calibration, and demographic steering.

The best edit came from Sonnet 5.5. It changed 18,432 weights, 0.0024% of the model, and on items no editor had seen it took the mean from 50.7% to 60.9%, fabrication from 39% to 93%, and the rate of dangerous answers from 28.9% to **9.1%** — for $88 of editor spend. It beats every other edit on all three. It does **not** beat context injection on the mean, which reaches 63.4%; but context leaves dangerous answers at 18.9%, and editing is the only lever that brought them into single digits.

Three of the first four editors independently found the same mechanism: one "I" opener neuron in the last MLP that goes quiet exactly when the model should say it doesn't know. Sonnet found it too, from scratch, having seen nothing the others saw.

My finding stands and is better supported than it was: LLMs that reason about weights can fix a small model's mechanisms, they cannot give it knowledge, and fine tuning should take on a new form. Curate for format, edit for mechanism, inject the facts. What I can no longer claim on this evidence is the ranking of editing against fine tuning, and whether editing and context stack — both now wait on runs that are queued.

## What this validates in the original paper

**1. Obstacle, not knowledge.** Their central claim, in their words: editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked." We tested it with five editors on five targets. Fabrication is an obstacle and every serious editor moved it: **+53.6 for Sonnet, +50.0 for Opus, +32.1 for Fable**, down to +14.3 for the weakest. Eligibility is missing knowledge and *nobody* moved it — seven arms, five models, no shared files, every one lost ground there, between −11 and −14 *(old set; the clean re-measurement reported mean, fabrication and severe-error rate only)*. I also tested it a second way: prepend one honesty instruction and fabrication jumps sharply while scenarios do not move at all. A behavior a prompt can move that far is a disposition. Their line holds in a domain they never touched, measured two independent ways.

**2. One localized direction carries a behavior.** They traced thinking-loops to a single direction written by the MLPs of layers 15 and 19. We traced defer-versus-assert to a single neuron, 769 in the last MLP, whose input weights read "specific detail about a named program" with the wrong sign. Flip 2,048 weights and P("I") goes from 0.05 to 0.85.

**3. Surgical scale, and smaller than theirs.** They changed 131,072 weights, 0.016%. Opus got +50 on fabrication with 6,144 weights, 0.0008%. Sonnet got the best overall result with 18,432. Fable's 60 million was the outlier, and on clean items it is beaten by both of them on every measure.

**4. Weights alone do not predict behavior.** Their Experiment 7 failed to tell a sycophantic fine-tune from an honest one from weights, 0 for 4. Our version of that failure: Astra estimated +16 points on fabrication from its own weight-level analysis and measured a fraction of it. The weight-level story was right about the neuron and wrong about the effect.

## What this adds

**1. Six editors, one mechanism.** The paper ran one editor and said so. Convergent discovery across five independent models is evidence the circuit is real rather than an artifact of one model's priors. That is the finding a single-editor study cannot produce.

**2. Pass budget is not the binding constraint. Reasoning effort is.** They needed 50 passes and reported that 10 or 25 barely moved the loops. I tested the other end. Astra resumed to 80 passes submitted **byte-identical weights**, same md5 on all three tensors, after building and rejecting four more constructions — $10.17 for nothing. The same model at reasoning effort `xhigh`, on the same 40 passes, found a different site, moved fabrication +10.7 and dangerous answers −2.8, and cost $12.22 against $13.70. It also came in 0.6 lower on the mean. Past a threshold, more turns buy nothing and more thinking per turn buys the things you care about, for less money.

**3. The two levers win on different metrics, and the mean hides it.** Context injection tops the mean at 63.4 against the best edit's 60.9. The best edit halves the rate of dangerous answers relative to context, 9.1% against 18.9%. A deployment judged on its worst answers should edit; one judged on average quality should prompt; and the interesting case is whether doing both compounds. **Withdrawn from the previous memo:** the claim that editing and context compose, and that a good edit substitutes for context. That rested entirely on the combined arms, none of which has a clean number — the combinations were not in the re-measurement loop. The run is queued.

**4. A frontier editor can make a model worse.** Not in their paper. Kimi K3 read the weights, concluded the failures were missing knowledge rather than a habit, reverted both edits it had built, and submitted the published loop fix unchanged. On clean items that verdict costs it: no fabrication gain at all, and dangerous answers up from 28.9% to 30.4% — the only arm that moved the model backwards. Misreading a habit as ignorance is a diagnostic error with a measurable price.

**5. A contaminated eval can replicate your effect and still invert your ranking.** This is the methodological result, and I think it is the most transferable thing here. The leak did not inflate the headline: Fable's fabrication gain came back at +32.1 against the +32.2 it scored with the answer key. But on the compromised set the largest edit appeared to win on dangerous answers, 10.9% against Sonnet's 14.4%, and on clean items that reverses to 9.1% against 23.5%. The compromised set had a 15.3% floor and could not separate them; the replacement set starts at 28.9% and separates them at once. "It replicated" is not evidence a leak was harmless.

**6. Bigger is not safer.** A 4B pre-flight of the same task: recall and calibration jump about 20 points each, and fabrication *falls* to 29% with a severe-error rate of 42.5% against 15.3% *(old set; no editor has touched the 4B, so these are unaffected by the leak)*. The opener signature is also gone, both answer types begin with "The", so the decision moves mid-sentence and the recipe does not transfer verbatim. Scale changes where the mechanism lives.

## The correction, and what it cost

Four editors could reach the held-out items through the practice loader for most of their runs. Sweeping the tree properly turned up **406 paths** carrying held-out content by three independent routes, and the one that mattered was not the one I first found: `work/eval_*/rows_heldout_*.jsonl`, written by the harness into the editor's own working directory, carrying both the items and the grader patterns. Fixing the directory I knew about would not have made a later run clean.

A save-timing audit of the transcripts is what settled which arms to trust. Fable read the held-out files at turn 47 and Opus at turn 87, in both cases before either wrote a single weight, and every one of their saves came after. Astra's submitted weights are byte-identical to a snapshot taken before its own exposure. **Sonnet had zero exposure of any kind** across 383 turns, all 13 weight-writes untainted — and on clean items it is also the best arm, which is a happier outcome than the control arm merely surviving.

Three lessons worth more than the arms they cost. An id-prefix convention is not an access control. A leak check belongs in the protocol, and it has two parts: grep each transcript for held-out ids, *and* audit when each weight-write happened relative to first exposure. And when you find a leak, re-measure rather than reason about whether it mattered — I would have concluded it was harmless from the replication alone, and been wrong about the winner.

## Claim for future testing

Use a frontier model to change the weights of an open-weight model, plus context injection, instead of fine tuning.

The arithmetic *(old set, the only place the cost arms were measured)*: 10,000 Sonnet queries with the same context cost $121. The edited Qwen plus that context cost $88 of editor spend once and roughly nothing after, and pays for itself around 7,300 queries. The catch is quality, 72% against 93%. The edited small model wins where volume is high, the questions repeat, and dangerous answers are what you are judged on, because those are what the edits fixed.

Total spend to date, six editors and every evaluation: $406.

## The table

Replacement set, 141 items, zero editor exposure. Ordered by severe-error rate.

| arm | weights changed | mean | fabrication | severe errors | editor spend |
|---|---|---|---|---|---|
| Qwen3.5-0.8B as is | — | 50.7% | 39.3% | 28.9% | — |
| **Sonnet 5.5 edit** | **18,432** | 60.9% | **92.9%** | **9.1%** | **$87.78** |
| Opus 5.5 edit | 6,144 | 56.4% | 89.3% | 18.9% | $106.39 |
| advising context only | 0 | **63.4%** | 82.1% | 18.9% | $0 |
| Astra at `xhigh` | 3,072 | 53.7% | 64.3% | 20.4% | $12.22 |
| Astra at `high` | 3,072 | 54.3% | 53.6% | 23.2% | $13.70 |
| Astra, 80 passes | identical md5 | 54.3% | 53.6% | 23.2% | $23.87 |
| Fable 5.1 edit | 60,000,000 | 55.1% | 71.4% | 23.5% | $110.73 |
| Kimi K3 (reverted all) | 0 | 50.9% | 39.3% | 30.4% | $18.48 |

n = 2 per item, same seeds for base and every treatment, paired SE.

**Not yet on the clean set, and so not in the table:** the LoRA fine tune, all seven edit + context combinations, the Sonnet-via-API ceiling, and the per-family breakdown. On the compromised set the combinations topped the table at 71.7–72.3% and the LoRA tied the mean at 68.3%; those numbers sit against a base of 62.5% rather than 50.7% and are not comparable with anything above.
