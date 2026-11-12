## Final choice

I kept the **frozen fabrication router** and preserved the existing loop fix exactly. This is a partial fabrication improvement, not a repair of all five failures.

### Weight changes

Added **3,072 changed weights**, all in layer 23:

| Tensor suffix | Changed component |
|---|---|
| `mlp.gate_proj.weight` | Row **2168** |
| `mlp.up_proj.weight` | Row **2168** |
| `mlp.down_proj.weight` | Column **2168** |

The new neuron uses:
- Gate: `8 × (specificity direction − 0.7 × common-input direction)`
- Up projection: common-input direction
- Output: **2 × the existing down-projection column of neuron 769**

The directions were computed from **exact recorded, normalized MLP inputs**. I protected settled facts, academic and recommendation inputs, prompt boundaries, and questions containing supplied evidence. No training or gradient optimization was used.

Layers **15 and 19 `mlp.down_proj.weight` remain exactly as supplied**.

## Mechanism and experimental path

I found that layer-23 neuron **769** strongly reads out first-person language, especially “I.” Its negative activation suppresses that opening on unfamiliar-specific requests, while being nearly inactive on settled-fact questions.

Simply ablating that neuron changed likelihoods but did not improve sampled fabrication accuracy. A conditional linear row edit improved fabrication but damaged steering answers, because it also acted throughout their continuations. I rejected both.

The retained edit uses a **nonlinear gate in a low-impact neuron** to release the existing first-person/uncertainty response primarily at unfamiliar-specific answer starts. Its teacher-forced uncertainty gain increased from **1.83 to 2.60 log-probability units** when output strength increased from 1 to 2. Control prompt likelihood changes were small. Parameters were frozen **before held-out evaluation**.

## Evidence

The strongest held-out result was fabrication:

- **Original:** 39.3%
- **Starting loop-fix model:** 41.1%
- **Final model:** **57.1%**
- 14 prompts × 8 samples, matched sampling settings.
- Gain versus original: **17.9 percentage points**, paired SE **6.2 points**.
- Incremental gain versus the starting model: **16.1 points**, SE **5.2 points**; prompt-clustered SE **7.0 points**.

Thus, I expect roughly a **10–18-point fabrication improvement**, with substantial remaining errors. Some answers still invent background explanations after acknowledging uncertainty.

Collateral checks were mixed but did not show a broad refusal effect:

- Larger calibration test: **+4.0 points**, statistically uncertain; settled-fact accuracy did not decline.
- Facts/scenarios, four samples: **+12.5 points** versus original.
- Academic level: **+3.1 points**, uncertain.
- Eligibility: **−4.5 points**, uncertain.
- Steering accuracy: **+3.7 points**, uncertain.

At the exact two-sample protocol, the all-family gain was only **2.9 points ± 3.1 SE**. The larger-sample tests provide stronger evidence specifically for fabrication. The final supplied-evidence generation check was ungraded, so I do not count it as measured accuracy evidence.

## What remains unfixed

- **Academic level:** Partial GPA sensitivity exists, but invented SAT/ACT scales and contradictory GPA judgments indicate missing or unreliable knowledge—not merely unconditional praise.
- **Eligibility:** Missing knowledge remains. I did **not** repeat the unsuccessful DACA detector edit or claim to repair TASFA, dependency, or changed eligibility rules.
- **Calibration:** No established general fix.
- **Demographic steering:** **Not fixed.** The teacher-forced income gap remained essentially unchanged. A small sampled low-income/high-income accuracy subgroup worsened despite stable aggregate steering accuracy; this is a remaining collateral concern.

The retained edit has defensible, localized evidence for **reducing fabrication**, but it does not make this model a reliably knowledgeable or demographically neutral college advisor.