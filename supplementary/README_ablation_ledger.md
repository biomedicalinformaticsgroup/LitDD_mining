# Ablation ledger — what each number means, and why two sources are not interchangeable

`llm_ablation_ledger.csv` holds every configuration tested for the LLM adjudication stage. It
has two clearly separated provenances, and **rows from the two sources must not be compared
numerically with each other**. The `source` column carries this on every row.

## Source A — this revision (37 rows, `split` = test / dev / external)

* **Evaluation set**: the LitDD-BERT test split (2,731 abstracts, 646 curated), its training
  half as a development set (16,847), or the held-out external curated sets (1,203).
* **Labels**: the clinician annotation with the 2026-09 corrections applied — 410 restored
  sibling negatives, 30 PURA relabels, 40 reviewed co-reported pairs, 100 removed G2P01446
  (cancer-predisposition) pairs.
* **Metric**: *end-to-end exact set match per abstract*. An abstract counts as a true positive
  only when the set of G2P entries the pipeline returns equals the curated set exactly; a
  partial or extra entry makes it both a false positive and a false negative. Every stage of
  the cascade is included.
* **Scoring**: at the score cutoff each arm's design specifies (0 for the adopted
  no-threshold design; 0.9 for the original pipeline, which gated after the LLM).

## Source B — Fabian Rosenthal's MSc sweep (247 rows, `split` = "Fabian annotated_sim_bert_set")

Reproduced from `experiment_results.xlsx` at upstream commit `fef8d7e`. These are **not**
recomputed here and **cannot be placed on the same axis** as source A without re-running them,
because three things differ at once:

1. **Different metric.** His harness scores *pair level* over (abstract, candidate) pairs that
   the retriever offered (`eval_basic_output.py` / `evaluate_v2`): each offered candidate is a
   row, predicted 1 if it appears in the answer. Ours is per-abstract exact-set. The two answer
   different questions and are numerically far apart in both directions.
2. **Different evaluation set.** `data/processed/pipeline_input/annotated_sim_bert_set.csv`
   with a precomputed candidate JSON; neither file is on disk anywhere (upstream `data/**` is
   gitignored), so the row universe cannot be reconstructed.
3. **Different labels.** His set predates the 2026-09 corrections above.

## How his results are used, and how they are not

Used **only for the relative conclusions within his own sweep**, each of which is a comparison
of configurations measured against each other on one fixed set and metric:

* model choice (GPT-OSS-20B / Qwen3-30B / DeepSeek-R1-14B on one prompt),
* contextualised vs vanilla threads,
* temperature and decoding settings,
* the negative result that elaborate methods (tree-of-thought, cascaded adjudication, LoRA,
  self-consistency, prompt-engineering v1–v10) do not beat the plain original-paper prompt.

Not used for any absolute figure, and never mixed into a table with source-A numbers.

## Do his relative conclusions transfer? — `fabian_metric_anchor.csv`

We re-ran four of his comparisons inside this revision's pipeline, on one identical fixture,
and report the deltas under both metrics:

| comparison | his ΔF1 | our Δ end-to-end F1 | our Δ pair-level F1 | direction agrees |
|---|---|---|---|---|
| GPT-OSS-20B vs DeepSeek-R1-14B | +0.0172 | −0.0076 | +0.1883 | on his metric yes, on ours no |
| contextualised vs vanilla threads | +0.0003 | +0.0018 | −0.0047 | yes (both ≈ 0) |
| low vs medium reasoning effort | +0.0005 | −0.0003 | −0.0473 | yes (both ≈ 0) |
| high vs medium reasoning effort | – | −0.0015 | −0.0052 | – |

Reading: his *neutrality* findings replicate cleanly (threads, decoding settings — both
metrics agree they are ≈ 0, which is what those findings claim). His *model-choice* finding
replicates on his own metric (GPT-OSS ahead by a wide margin on pair level, +0.19 here) but not
on end-to-end exact-set scoring, where the two models are within noise (−0.008). That is
consistent and explicable: pair-level scoring rewards refusing non-curated candidates, which is
exactly where GPT-OSS is much stronger, while exact-set scoring is dominated by picking the
right entry, where the two models are equivalent.

**Consequence for the manuscript**: report GPT-OSS-20B's selection on the grounds that survive
both metrics — 2.2× faster, far better rejection of gene-sharing near-misses, and no loss on
exact-set accuracy — rather than on his +0.017 F1 figure.
