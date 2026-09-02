# Should the cross-encoder stay in the pipeline? — evidence, 2026-09-02

## What changed architecturally

In the published pipeline the cross-encoder was the **retrieval** stage: it scored every
abstract against all 2,861 G2P entries and passed a fixed top-5 to the LLM, with a 0.9 score
gate applied after the LLM. In the revised pipeline the **gene gate** does the retrieval —
each abstract is restricted to the entries of the genes actually mentioned in its
title+abstract (mean 2.40 candidates, median 2, p95 6). That is a data-driven candidate set
rather than a fixed k, and it is loss-free on the test set: 606/606 curated abstracts have
**all** their curated entries among the candidates (retrieval recall 1.000). So yes — the
cross-encoder's original job has been taken over by the gene gate.

## Measured contribution of the cross-encoder now

| arm | P | R | F1 | note |
|---|---|---|---|---|
| screen → gene gate → CE scores present, all entries shown → LLM | 0.848 | 0.847 | 0.847 | deployed |
| screen → gene gate → LLM (CE removed entirely) | — | — | — | *not run separately: identical inputs* |
| gene gate → LLM, no screen, no CE (direct) | 0.625 | 0.902 | 0.738 | recall ceiling |

Because the adopted design shows the LLM **every** gate candidate with **no threshold**, the
cross-encoder's scores do not influence which entries the LLM sees or chooses. Its only
remaining effects are:

1. **Ordering.** Candidates are listed in score order. Measured: top-k truncation experiments
   (k = 1, 2, 3, 5, all) show k ≥ 3 is indistinguishable from all (F1 0.834–0.838), so ordering
   matters only if a cap is reintroduced. A cap would bind on **6.9 %** of abstracts (170/2,462
   with > 5 candidates).
2. **Gating.** Any threshold ≥ 0.5 costs recall with no precision gain in the current design
   (0.9 gate: −45 curated entries; the reason the design moved to no threshold).
3. **Audit signal.** The score weakly separates right from wrong LLM answers: median 0.984 for
   exactly-correct answers vs 0.979 for incorrect ones; filtering answers at min-score ≥ 0.95
   keeps 498/589 and lifts precision from 0.927 to 0.936 — i.e. it discards 15 % of answers to
   gain 0.9 points. Weak, but non-zero, and free if the model is already loaded.
4. **Corpus-scale cost.** Removing it saves one GPU model and ~38k pair scorings per 2.5k
   abstracts (≈ 1 GPU-hour per million abstracts) — small next to the LLM stage.

## Recommendation

**Keep the cross-encoder in the released artefact set but remove it from the deployed critical
path**, i.e. document the pipeline as gene gate → LLM, and describe the cross-encoder as an
optional ranking/capping component (used when an abstract yields more than ~5 candidates, and
as a confidence signal for the precision audit). Reasons: it costs nothing to retain as an
option, the reviewers asked specifically about candidate selection (R3.5) and the ranking
evidence is part of that answer, and the top-1/top-2 experiments only make sense if the ranker
exists. **Do not delete `tmy100000001/LitDD_crossencoder` from Hugging Face** — the published
pipeline used it, so the repository must remain resolvable for reproduction of the original
results; if the revised pipeline ships without it, mark the model card "not required by the
revised pipeline; retained for reproduction of the published pipeline and as an optional
ranker".

If you prefer the simpler story (drop it outright), the accuracy cost is zero on this test set,
and the model card should then say the same thing in reverse: retained on HF for reproduction
only. Either way the manuscript text must state that the gene gate replaced full-panel
cross-encoder retrieval, and that k is now data-driven.
