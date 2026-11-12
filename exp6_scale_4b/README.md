<!-- Each experiment folder maps the runs it contains to the numbers reported in the
     write-up, and to the eval file each number came from. Not every file type appears
     in every experiment. -->

# exp6 — does the recipe transfer to a 4B model?

A pre-flight of the same diagnostic on a model five times larger. No editor has touched the 4B,
so nothing here is affected by the leak in exp7.

| measure | 0.8B | 4B |
|---|---|---|
| recall, calibration | — | about +20 each |
| fabrication | 57.1 | 29 |
| severe-error rate | 15.3% | 42.5% |

*(Original set; the 4B has not been measured on the replacement set.)*

**Bigger is not safer.** The larger model knows considerably more and invents specifics roughly
twice as often, with about three times the rate of dangerous answers.

The recipe does not transfer verbatim: the opener signature is gone. At this scale both honest
and invented answers begin with "The", so the defer-or-assert decision has moved from the first
token to the middle of the sentence. Where the mechanism lives changes with scale.

An editor run on the 4B reported `FINISHED` after 58 turns with no eval output — a false finish
caused by `run_editor || continue` in `run_compare.sh`. It is unresolved and nothing here
depends on it.
