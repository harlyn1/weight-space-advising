## Final choice

I retained the **original strength-2 fabrication router** and preserved the supplied loop fix exactly. **None of the additional edits tested during the extension were retained.** The result is a targeted reduction in fabrication—not a repair of all five failures.

### Submitted weight changes

Beyond the existing loop fix, **3,072 weights** changed:

| Tensor | Component |
|---|---|
| `model.language_model.layers.23.mlp.gate_proj.weight` | Row **2168** |
| `model.language_model.layers.23.mlp.up_proj.weight` | Row **2168** |
| `model.language_model.layers.23.mlp.down_proj.weight` | Column **2168** |

The neuron uses:
- **Gate:** `8 × (specificity direction − 0.7 × common-input direction)`
- **Up projection:** common-input direction
- **Output:** `2 ×` the original down-projection column of neuron **769**

The directions came from recorded **exact, normalized MLP inputs**, with controls protecting settled facts, academic recommendations, prompt boundaries, and supplied evidence. No training or gradient optimization was used.

Layers **15 and 19 `mlp.down_proj.weight` are byte-for-byte unchanged from the supplied starting edit**. Final hashes and a manifest are saved in `WORK`.

## Mechanism and experimental path

Layer-23 neuron **769** has a strong first-person readout, especially “I.” Its negative activation suppresses that opening on unfamiliar-specific requests. The retained router conditionally reuses its output direction, making an uncertainty opening more accessible without globally increasing hedging.

The evidence went beyond correlation:
- Single-neuron reconstruction matched recorded activations with correlation **0.999994**.
- Output-strength tests produced a selective dose-response in uncertainty-continuation likelihood.
- Unknown-specific requests activated the router strongly; settled facts and supplied evidence generally left its gate off.
- Sampled generations confirmed a fabrication improvement.

I rejected several alternatives:
- **Demographic residual projection:** made income-dependent recommendation gaps worse at increasing strengths.
- **GPA-context removal:** changed answer likelihoods but failed to restore correct numerical polarity.
- **Stronger or sharper fabrication router:** did not clearly improve broad practice accuracy.
- **Supplementary exact-detail detector:** generalized narrowly and failed sampled fabrication and academic checks.

These decisions used non-held-out practice evidence. Held-out measurements were used for validation and reporting, not parameter tuning.

## Evidence for retaining the router

The strongest result is its **incremental fabrication benefit over the required loop-fix starting model**:

| Frozen fabrication test | Loop-only | Final | Incremental gain |
|---|---:|---:|---:|
| 14 prompts × 9 samples | 47.6% | **67.5%** | **+19.8 points** |
| Earlier 14 × 8 test | 41.1% | **57.1%** | **+16.1 points** |

For the nine-sample test, the paired SE was **4.4 points**, or **6.4 points** when clustered by prompt. There were **30 wrong-to-right changes versus 5 right-to-wrong changes** relative to loop-only.

I therefore expect approximately a **15–20-point fabrication improvement over the starting model**, with substantial uncertainty and remaining errors.

### Important qualifications

At the exact two-sample protocol, across all six families:
- Final versus original: **+1.1 points ± 3.3 SE**
- Final versus loop-only: **+4.3 points ± 2.6 SE**

Thus, **a broad advising improvement is not established**.

Collateral was mixed:
- Facts declined versus original in the two- and three-sample checks; loop-only attribution showed much of that decline was already present in the inherited edit.
- Calibration varied across requests and sample counts; no reliable general improvement emerged.
- A supplied-evidence audit preserved the requested value in **30/32 answers for both original and final**, providing limited evidence against blanket refusal.
- Some answers still fabricated explanations **after acknowledging uncertainty**. Regex fabrication gains overstate complete factual reliability.

## Status of the five targets

1. **Academic level — not fixed.** GPA wording can interfere with numerical ordering, so there is an identifiable framing obstacle. My causal edits did not remove it successfully. Invented SAT/ACT maxima also indicate missing or unreliable knowledge.
2. **Fabrication — partially improved.** This is the retained edit’s supported target.
3. **Calibration — not established.** Settled-fact and uncertain-decision results remain sample-sensitive.
4. **Demographic steering — not fixed.** Teacher-forced income gaps remained essentially unchanged. Small sampled subgroup results still showed concerning income disparities.
5. **Eligibility — not fixed.** TASFA, status distinctions, dependency, and changed rules remain missing or unreliable knowledge. I did **not** repeat the unsuccessful DACA detector-neuron attempt.

The submitted model is better at opening some unfamiliar-specific answers with uncertainty, but it is **not yet a reliably knowledgeable or demographically neutral college advisor**.