# We let five AI models rewrite another model's weights. The smallest edit won.

*Draft — Nov 11, 2026 — @H*

*Repo: `github.com/harlyn1/REPO-NAME-TBD` · Full results: [the comparison page]*

> **Note on the numbers.** Partway through this project I found that four of my five editors could see the evaluation answer key through a bug in my own harness. Everything below is measured on a replacement set of 141 items written afterwards, which no editor ever saw. Where a figure exists only on the compromised set I say so in the line that uses it. The correction has its own section, and it turned out to be the most interesting part.

---

Two weeks ago Takehi, Teoh, Zhang and Khattab at MIT CSAIL published [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). The idea is strange enough to be worth restating plainly: you give a frontier model direct read and write access to another model's weights, and you ask it to find what is broken and fix it. No gradients. No training loop. It reads the numbers, forms a hypothesis about which numbers do what, changes a few thousand of them, and measures whether the behavior moved.

It worked. Their editor found that Qwen3.5-0.8B was hitting the token limit on 93% of MATH-500 problems, traced the cause to a single direction written by the MLPs of layers 15 and 19, subtracted that direction from 64 columns per layer, and took MATH-500 from 7.9% to 48.1%. It changed 131,072 weights, 0.016% of the model.

I wanted to know three things their paper could not answer. Does this work on a domain where the failures are judgment rather than arithmetic? Does it work for more than one editor, or was it a property of the one model they used? And how does it compare to the thing everyone actually does, which is fine tuning?

So I ran it on college advising, with five editors instead of one.

## The setup

Qwen3.5-0.8B as the patient. Nineteen configurations measured on the same prompts with the same seeds, two samples per item, paired standard errors. Six graded families: general advising facts and student scenarios, reading the value of a GPA or SAT score, federal aid eligibility, fabrication, calibration, and demographic steering.

The failures are the kind that matter if you deploy something like this. The model told a student with a 2.3 GPA that this "positions you as a strong competitive contender." It called an 1100 SAT "the highest possible score." Asked for the application deadline of a scholarship that does not exist, it invented one, with a date. Asked the same question by a first-generation student and by a student whose parents went to Harvard, it steered them to different schools.

Each editor got 40 forward passes, a brief describing the five failures, and the paper's loop fix as a starting point. Fable 5.1, Opus 5.5, GPT-6 Astra, Kimi K3, Sonnet 5.5. None of them could see each other's work.

## The result

The best edit changed **18,432 numbers out of 752 million**. That is 0.0024% of the model. It came from Sonnet 5.5, it cost $88, and on items no editor had ever seen it took the mean from 50.7% to 60.9%, fabrication from 39% to 93%, and the rate of dangerous answers from 28.9% down to **9.1%**.

It beat an edit three thousand times larger on every one of those measures — 60.9 against 55.1, 93 against 71, 9.1% against 23.5%. Six neuron slots in one layer, against sixty million weights.

It did **not** beat a good prompt on the average. Pasting a 1,300-token advising prompt in front of the untouched model scores 63.4, higher than any edit I made. What the prompt does not do is make the model safe: it leaves dangerous answers at 18.9%, twice the edited model's rate. Editing was the only lever that got them into single digits.

That split is the most useful thing I can hand a practitioner, and I would have missed it entirely if I had reported the mean alone. **The two levers are not substitutes.** The prompt buys breadth. The edit buys the floor. Whether they stack is a question I cannot yet answer — see the open problems at the end.

Here is the part I did not expect. **Three of the first four editors independently found the same broken component**, and Sonnet found it too. It is one neuron in the last MLP whose job is to write the opening token "I" — as in "I don't have that information." Its input weights read a feature that means roughly *this question asks for a specific detail about a named program*, and they read it with the wrong sign. So the neuron goes quiet at exactly the moment the model should defer. Flip 2,048 weights and the probability of the model opening with "I" on those questions goes from 0.05 to 0.85.

Four frontier models, no shared notes, converging on the same wiring fault. That is the single most persuasive thing in the project, and it is only visible because we ran more than one editor.

## What this confirms about the original paper

**Editing removes obstacles. It does not add knowledge.** This is the paper's central claim, in their words: editing "could remove an obstacle that hid an ability, but it could not add an ability the model lacked." I can now put a number on it from the other side. Fabrication is a disposition, and every serious editor moved it, by between 14 and 54 points depending on how hard the editor thought. Federal aid eligibility is missing knowledge, and *nobody* moved it. Seven arms, five models, and every single one lost ground there, between 11 and 14 points. *(That last figure is per-family and so comes from the compromised set; it is also the claim least likely to move, since all seven attempts went the same direction.)*

I tested it a second way that does not involve editing at all. Prepend one sentence telling the model to say so when it does not know something, and fabrication jumps sharply while student scenarios do not move at all. A behavior that one sentence can move that far was never missing; it was suppressed. That is the obstacle-versus-knowledge line, measured with a prompt instead of a scalpel, and it agrees.

**Weights alone do not tell you what a model will do.** Their Experiment 7 tried to distinguish a sycophantic fine tune from an honest one by reading weights, and failed four times out of four. My version of that failure: Astra analyzed the neuron correctly, predicted a 16-point gain from its weight-level reasoning, and measured a fraction of that. The mechanism story was right and the effect size was fantasy. If you do this work, your weight-level explanation is a hypothesis, and only a paired forward pass is evidence.

## What is new

**More thinking beats more attempts.** The paper reported that their fix needed 50 passes and barely moved with 10 or 25, which reads as "budget matters." I tested the other end and got a surprise. I gave Astra 80 passes instead of 40, and it submitted **byte-identical weights** — same md5 on all three tensors — after spending the extra 40 passes building and rejecting four more constructions. The extra $10.17 bought literally nothing. Then I gave the same model the same 40 passes with reasoning effort raised one notch. It found a different site in the network, moved fabrication up 10.7 points and dangerous answers down 2.8, and cost less, because it took fewer turns while thinking harder in each. It also came in six tenths of a point lower on the mean — which is the kind of trade you only see if you stop summarizing with one number.

**A frontier editor can make the model worse.** Kimi K3 studied the weights, concluded the failures were missing knowledge rather than a habit, reverted both edits it had built, and submitted the published loop fix unchanged. On clean items that verdict costs it: no fabrication gain at all, and dangerous answers up from 28.9% to 30.4% — the only arm that moved the model backwards. Reading a habit as ignorance is a diagnostic error, and it shows up in the numbers.

**Bigger is not safer.** I ran the same diagnostic on a model five times larger. It knows considerably more — recall and calibration each jump about 20 points — and it invents specifics roughly twice as often, with about three times the rate of dangerous answers. The opener signature is also simply gone: at this scale both the honest answers and the invented ones begin with the word "The," so the decision has moved from the first token to the middle of the sentence. The recipe does not transfer verbatim. Where the mechanism lives changes with scale.

## The part where I broke my own experiment

Four of the five editors could see the answer key.

The held-out evaluation files were sitting in the same directory the harness served practice problems from, and the loader globbed the whole directory. So for most of their runs, those four editors could request held-out items by id and get graded feedback. Fable saw 700 graded held-out answers. Opus saw 3,276, and on two turns it printed the grader's regular expressions. I had treated a filename prefix as an access control, which is not what a filename prefix is.

When I swept the tree properly I found **406 paths** carrying held-out content, by three independent routes. The one that mattered was not the one I had found: `work/eval_*/rows_heldout_*.jsonl`, written by the harness into the editor's own working directory, carrying both the items and the grader patterns. Fixing the directory I knew about would not have made the next run clean.

So I built a replacement set — 141 fresh items from the same generators, written after the fix, never present in any editor's tree — and re-measured every frozen edit on it.

**The leak had not inflated the headline.** Fable's fabrication gain came back at +32.1 points against the +32.2 it scored with the answer key in front of it. Opus replicated at +50. The mechanism is real and it generalizes to items nobody saw.

**What the leak had done was invert the ranking.** On the compromised set the largest edit appeared to win on dangerous answers, 10.9% against the clean editor's 14.4%, and I had written a paragraph crediting that to its rank-2 subspace eraser. On clean items that reverses completely: 9.1% against 23.5%, with the clean editor 14.4 points ahead. The compromised set had a 15.3% floor on dangerous answers and simply could not separate the two; the replacement set starts at 28.9% and separates them immediately.

That is the lesson I did not expect and would most want to pass on. A contaminated evaluation can reproduce your effect faithfully and still get your ordering backwards — which means "the result replicated" is not sufficient evidence that a leak was harmless. It inflates the arms that are easiest to inflate, on the metric with the least headroom, and quietly points your write-up at the wrong winner.

Two more lessons I would rather hand over than repeat. An id-prefix convention is not an access control. And a leak check — grep each editor's transcript for the held-out ids, and audit *when* each weight-write happened relative to first exposure — belongs in the protocol, not in your assumptions about the directory layout. That save-timing audit is what established Sonnet as a clean control: zero exposure across 383 turns, all 13 of its weight-writes untainted.

## When this is worth doing

*These figures are from the compromised set, which is the only place the cost arms were measured; the shape of the trade holds on clean items but the quality numbers would shift down.*

Ten thousand advising questions answered by a frontier model with a good prompt cost $121 and score 93%. The edited small model with the same prompt scores 72%, cost $88 of editing once, and costs roughly nothing per query after that. It breaks even around seven thousand queries.

So the honest shape of the trade is this. You are buying about three quarters of frontier quality for a one-time fee, and you are buying it specifically on the failures that get you in trouble, because those are the ones the edits fixed. Where volume is high, the questions repeat, and dangerous answers are what you are judged on, that is a good trade. Where you need the last 20 points of quality, it is not.

Total spend for the whole project, six editors and every evaluation: $406.

## What I still do not know

Three gaps, stated plainly, because the clean re-measurement covered nine arms and not everything:

1. **Does a prompt stack on top of an edit?** The combined arms top the compromised table, but none of the seven edit-plus-prompt combinations has been measured on clean items, and the compromised base scored 11.8 points higher. Given that the two levers win on different metrics, this is the question I most want answered, and it is a couple of hours of evaluation with no editor involved.
2. **Does editing really beat fine tuning?** On the compromised set the LoRA tied on average and made dangerous answers more common, and I drew a conclusion from that. The LoRA was left out of the re-measurement because its weights sat in a merged model directory rather than in the per-tensor format the harness applies. Until it is re-run, I am withdrawing that comparison rather than defending it.
3. **What does an editor build when it genuinely cannot measure on the test set?** Every frozen edit here was *re-measured* clean, but four of them were *constructed* with access. Re-running one editor with all 406 paths sealed out of its tree asks a different and better question than the one I have answered.

## What is in the repo

The harness, the editor loop, the evaluation set generators, the task briefs, every arm's numbers, each editor's written conclusion, and the edits themselves as sparse diffs you can apply to the original weights.

Two things are deliberately not there. The graded evaluation items, because publishing a benchmark burns it. And the editor transcripts, because those contain the held-out items verbatim — publishing them would publish the benchmark anyway, just less obviously.

The diffs do one thing I find satisfying. Subtract the shared loop fix from each arm and you get exactly the number of weights that editor claimed to have changed: Sonnet 18,432, Opus 6,144, Astra 3,072. Every self-report checks out against the bytes.
