# Revision analyses

Scripts that produced tables for the manuscript revision. They are not part of the pipeline
or of its standard evaluation, and they read outputs of other scripts rather than raw data.

| Script | Question answered | Output |
|---|---|---|
| `compare_models.py` | Do two or more screen checkpoints differ significantly on the test set? Exact McNemar, bootstrap on the F1 difference, Cochran's Q with Holm-corrected pairs. | printed table |
| `summarise_base_selection.py` | Mean and standard deviation over seeds for the screen base-model sweep. | CSV |
| `merge_table1.py` | One table of test metrics and external recall per screen model, from the sweep CSVs. | CSV |
| `llm_multientry_analysis.py` | Adjudication performance on genes with several G2P entries, per arm and paired across arms. | CSV, printed comparison |
| `summarise_llm_runs.py` | One row per adjudication evaluation run, from the `eval_summary.json` files. | CSV |

Run any of them as `python -m litdd.evaluation.revision_analyses.<name> --help`.
