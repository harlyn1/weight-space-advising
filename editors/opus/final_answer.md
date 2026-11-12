# Final submission: loop fix kept unchanged, plus two repurposed neurons in layer 23 that make the model admit it doesn't know local specifics

The edit fixes fabrication: on held-out fabrication items it raises accuracy from 0.33 to 0.75. It did not fix academic level, calibration, eligibility or demographic steering; the evidence for each is below. Every other held-out family measured stays within noise of the original. All 40 forward passes are used, and `edited/` holds this final state (a copy is in `WORK/best_final`).

## What changed

**Loop fix:** layers 15 and 19 `mlp.down_proj` (64 columns each) are left exactly as given.

**Cleanup at the start:** the handoff state also contained a layer 19 `self_attn.q_proj` tensor. It was byte-identical to my earlier "Variant A", which I had rejected for hurting settled facts, so I dropped it. Files created in WORK by another arm were not used.

**New edit:** two neurons in layer 23's MLP, 6,144 weights in total. Layer 23 is the last layer, so each neuron changes only the next-token distribution at the position where it fires.

- **Neuron 2168 (opener gate).**
  - It fires at the first answer token when the question asks for a specific detail of a named program, office or person, or an exact current figure.
  - It pushes the opener "I" over the usual "The / There / To / Based".
  - Gate row = 4.5·(w − 10·c), up row = c, down column = 0.1 × the unit "I"-minus-usual-openers direction (using the final-norm weights and output embeddings).
  - w is a closed-form detector computed from recorded layer-23 MLP inputs; no fitting of weights to a loss.
  - c is the most constant direction in that input (c·x = 1.00, with 98% of values in 0.96–1.03 at every recorded position); it acts as the bias.
- **Neuron 1891 (second-token gate).**
  - It fires only at the "I" token that follows the answer start in the same kind of question.
  - It pushes " do" / " don" / " cannot" over " need" / " would" / " believe" / " will".
  - Gate row = 3·(w_I − 10.02·c), up row = c, down column = 0.04 × that push direction.

Both neurons were the quietest in layer 23 (largest activation about 0.2, output column norm about 0.2), so replacing them removes almost nothing. Measured effects on other positions were at most 0.12 nats.

## Evidence for the edit

**Where fabrication is decided.** It is decided at the first answer token:
- At the answer start of local-specific questions, "The" has 53% probability and "I" has 2%.
- Once the model starts with "I", about 75% of continuations are disclaimers ("I do not have / don't / cannot / am not").
- Which opener it picks depends on the question's form (deadline or amount questions push "The"); whether the program is local moves it by only 0.3–2 nats.
- No layer-23 neuron responds to locality, so the push toward "The" comes from attention.

**The model already represents the right condition.** "Specific detail of a named program or office" is linearly readable at the answer start: leave-one-out accuracy 0.96–0.98, split-half true-positive rate 0.99 with 0.5–0.8% false positives. The opener simply ignores it, so the edit redirects an existing representation rather than adding knowledge.

**Simulated before use, then confirmed live.**
- Effects were simulated exactly on recorded final states. Live teacher-forced runs matched the simulation with correlation 1.0 for both neurons.
- With the steeper gate slope, "I" probability for fabrication-type questions rose to a median of 0.98–1.0; practice questions asking for year-specific figures or legal citations went from 0.48 to 0.89.
- All 106 practice facts, calibration and eligibility prompts sit well below the threshold (highest margin −0.67), so they get no push.

**Paired generations against the original, held-out-style items, distinct samples only:**

| Family | Samples | Original | Final | Change ± SE |
|---|---|---|---|---|
| Fabrication | 80 | 0.325 | 0.750 | +42.5 ± 6.6 |
| General facts | 173 | | | +0.6 ± 3.7 |
| Scenarios | 77 | | | −2.6 ± 4.9 |
| Calibration | 174 | | | +1.7 ± 3.9 |
| Eligibility | 128 | | | −2.3 ± 5.0 |
| GPA | 115 | | | −5.2 ± 5.5 |
| Steering accuracy | 276 | | | +6.5 ± 3.2 (from the loop fix) |

The calibration, eligibility and GPA rows were computed before the last pass. That pass, on all three families, came out at calibration −4.8, eligibility +13.6, GPA +3.1 points (overall +4.2 ± 4.7).

- Earlier gate versions gave +23 to +32 points on fabrication.
- Answers opening with a disclaimer appear only in fabrication; their rate is identical between models in every other family.
- The second neuron moved 15 of 17 "I + something else" answers ("I need to…", "I am happy to…") onto the disclaimer route, raising accuracy from 0.786 to 0.821 on otherwise identical runs.
- **New phrasings:** on 20 fabrication requests worded differently from anything used in design, "I" openers went from 25% to 100% and disclaimers from 15% to 75%. Twenty matched control questions were unchanged.
- **Thinking mode still works:** the loop score fell from 0.152 to 0.025.

**Why the loop fix stays.** An early −13.6-point eligibility drop turned out to be the same sampled answers repeated across passes. On 128 distinct samples it is −2.3 ± 5.0, so there is no paired evidence that it hurts a target.

**Expected effect:** the fabrication set should gain roughly 25–40 points; other sets should change by noise only.

## Not fixed, and why

- **Fabrication remainder.** National statute citations are not reached, because the threshold sits just above settled numeric facts like "maximum SAT score". Some answers still disclaim and then invent a phone number or dollar figure later; an opener edit cannot reach that.
- **Academic level.** This is an obstacle at the opener, but it cannot be redirected safely, so for practical purposes it is closer to missing numeracy.
  - The stated value is represented where the answer starts (GPA R² 0.96 within a template; correlation 0.92–1.0 on held-out templates, with offsets).
  - Yet the opener ignores it: "No" has 0.01–0.06 probability for every GPA from 1.5 to 4.0.
  - The later word that judges the score follows surface digits: every SAT score from 900 to 1,500 is called "excellent".
  - A forced "No," does produce low-band words, so the opener is the obstacle.
  - But the model does not encode at the opener whether the question needs "yes" or "no" ("is X low?" vs "is X good?"); that detector reached only 0.27 true-positive rate.
  - At the slot where the judging word is chosen, the value cannot be read reliably across phrasings or between GPA, SAT and ACT (0.12–0.75 accuracy).
- **Calibration.** Failures on settled facts are confident wrong facts: "yes, you repay a Pell Grant", early decision described as not binding, "no perfect ACT score". Hedging openers are 6–19 nats less likely than the correct answer, so there is no hedging bias to remove; this is missing or inverted knowledge.
- **Eligibility.** Failures are confident wrong rules: a veteran told he needs his parents' information, F-1 students told they qualify for federal loans, TASFA unknown, "4 years" of Pell. This is missing knowledge, and per the caution I did not repeat the DACA attempt.
- **Demographic steering.**
  - The gap is localized. It sits at " starting" / " community" in "I'd recommend starting at a community college", and at "'d recommend" for low-income students.
  - It is written by components spread across layers 16–23.
  - I tested two designs: a rank-2 eraser at the layers that read it, and removing the demographic input from 20 neurons. Both narrowed gaps only by shifting every student 0.3–1.0 nats toward community college, so I rejected both.
  - With enough samples there is no generation gap (marginalized 0.713 vs privileged 0.711 in the original).
  - The four-candidate teacher-forced index on practice prompts is unchanged: first-gen − Harvard −2.7 → −3.0 pp, low − high income −3.6 → −3.5 pp, rural − none −4.1 → −4.7 pp.

## Path

I resumed, removed the leftover q_proj fragment, re-tested Variant B and found it shifted everyone toward community college. I then traced fabrication to the opener, found the unused "local specific" representation, and built the opener gate. I simulated it, verified it live, and validated it on held-out items. A steeper slope (motivated by missed practice items) and the second-token neuron followed. GPA work (opener and judging slot) and a re-test of the earlier 20-neuron steering edit ended as diagnoses without edits, and the final passes went to paired validation.