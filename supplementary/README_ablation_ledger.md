# Ablation ledger: what each column means

`llm_ablation_ledger.csv` records every configuration evaluated for the disease-adjudication
stage during the revision. All rows share one evaluation basis, so any two may be compared
directly.

## Evaluation basis (identical for every row)

* **Sets**: the annotated test split, its training half used as a development set for model
  and prompt selection, and the held-out external curated sets (pre-mined DDG2P, HPOA,
  ClinGen). The `split` column says which.
* **Labels**: the clinician annotation with the reviewed corrections applied (restored sibling
  negatives, relabelled and removed entries), as recorded in `data/annotation_corrections.csv`
  and applied by `litdd/evaluation/apply_annotation_corrections.py` and
  `complete_test_pairs.py`.
* **Metric**: end-to-end exact set match per abstract. An abstract is a true positive only
  when the set of G2P entries the pipeline returns equals the curated set exactly; a partial
  or extra entry makes it both a false positive and a false negative. Every stage of the
  cascade counts.
* **Scoring cutoff**: no arm applies a score threshold (`design_cutoff` 0); the `pre-LLM`
  rows applied a candidate-score gate before adjudication and are listed for completeness.
  `CE` in a description is the cross-encoder ranking stage of the submitted pipeline, which
  the released pipeline does not use.

## Columns

| column | meaning |
|---|---|
| `run` | run identifier |
| `split` | test / dev / external |
| `description` | the configuration in words |
| `design_cutoff` | score cutoff applied when scoring this arm |
| `TP FP FN TN` | end-to-end confusion matrix over the whole evaluation set |
| `precision recall f1` | end-to-end exact-set metrics |
| `pair_level_*` | the same run scored per labelled (abstract, entry) pair |
| `no_match_rate` | share of abstracts the adjudicator declined to map |
| `multi_gold_exact` | exact-set accuracy on abstracts with more than one curated entry |
| `share_gene_exact` | exact-set accuracy on abstracts whose candidates include several entries of one gene |
| `rows_per_s` | adjudication throughput on one H100 |

## Released configuration

The rows named `*_gatedtb_allelicspectrum_allpanels_hgncgate` are the released pipeline
(HGNC-identifier gene gate, contextualised threads with same-gene entries from every panel,
the decision rubric in `litdd/pipeline/prompts/decision_rubric.txt`, no score threshold) on
the test, development and external sets. Other rows are configurations that were evaluated
and not adopted; the code paths that produced them are not part of this repository.

## Other tables

`stage_confusion_matrices.csv` holds the per-stage confusion matrices of the released
pipeline on the test set (`litdd/evaluation/stage_confusion_matrices.py`).
