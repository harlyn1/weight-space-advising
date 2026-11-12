<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp8 — the mechanism four editors found independently

One neuron in the final MLP writes the opening token "I", as in "I don't have that information".
Its gate row reads a feature meaning roughly *this question asks for a specific detail about a
named program* — and reads it with the wrong sign. The neuron therefore goes quiet at exactly
the moment the model should defer, and the model asserts instead.

Flipping 2,048 weights takes P("I") on those questions from **0.05 to 0.85**.

Fable, Opus, Astra and Sonnet each located this independently, with no shared notes. Kimi read
the same failures as missing knowledge, declined to edit, and produced the only arm that left
the model worse than it started — a diagnostic error with a measurable price.

`prompt_feature_audit.md` is the brief used for the feature audit.

**Open.** The opener probe needs re-running at the mid-sentence decision point rather than the
first answer token, with larger probe sets: decodability probes are gated at n ≥ 100 after a
first attempt returned R² of −708 at n ≈ 30. At 4B the opener signature is absent entirely
(exp6), so the probe has to move with it.
