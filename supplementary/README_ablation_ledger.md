# Ablation ledger — what each column means

`llm_ablation_ledger.csv` records every configuration evaluated for the LLM disease-adjudication
stage. All rows share one evaluation basis, so any two may be compared directly.

## Evaluation basis (identical for every row)

* **Sets**: the LitDD-BERT test split (2,731 abstracts, 646 curated), its training half used as a
  development set for model and prompt selection (16,847 abstracts), and the held-out external
  curated sets (premined / HPOA / ClinGen; 1,203 abstracts). The `split` column says which.
* **Labels**: the clinician annotation with the reviewed corrections applied — restored sibling
  negatives, the PURA relabelling, the reviewed co-reported pairs, and the removal of a
  cancer-predisposition entry that is out of scope for DDG2P. Applied by
  `litdd/evaluation/apply_annotation_corrections.py` and `complete_test_pairs.py` from
  `data/annotation_corrections.csv`.
* **Metric**: end-to-end exact set match per abstract. An abstract is a true positive only when
  the set of G2P entries the pipeline returns equals the curated set exactly; a partial or extra
  entry makes it both a false positive and a false negative. Every stage of the cascade counts.
* **Scoring cutoff**: each arm is scored at the cutoff its design specifies — 0 for the adopted
  no-threshold design, 0.9 for the original pipeline, which applied its score gate after the LLM.
  The `design_cutoff` column records this.

## Columns

| column | meaning |
|---|---|
| `run` | run identifier; outputs under `revision/llm_eval/runs/<run>/` |
| `split` | test / dev / external |
| `description` | the configuration in words |
| `design_cutoff` | score cutoff applied when scoring this arm |
| `TP FP FN TN` | end-to-end confusion matrix over the whole evaluation set |
| `precision recall f1` | end-to-end exact-set metrics |
| `pair_level_*` | the same run scored per labelled (abstract, entry) pair, for readers who prefer a pair-level view |
| `no_match_rate` | share of abstracts the adjudicator declined to map |
| `multi_gold_exact` | exact-set accuracy on abstracts with more than one curated entry |
| `share_gene_exact` | exact-set accuracy on abstracts whose candidates include several entries of one gene (the allelic-series case) |
| `rows_per_s` | adjudication throughput on one H100 |

## Reproducing a row

```bash
python litdd/evaluation/build_ablation_ledger.py --out_csv supplementary/llm_ablation_ledger.csv
```

Per-stage confusion matrices for the deployed cascade are in
`stage_confusion_matrices.csv` (`stage_confusion_matrices_with_crossencoder.csv` for the variant
that retains the cross-encoder as a ranker), and the comparison against the published pipeline is
in `original_vs_new_pipeline.csv`.
