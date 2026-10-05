# Training

Scripts that build the screen's training data and train the screen. Run them as modules from
the repository root, for example `python -m litdd.training.final_traintest_dataset --dry_run`.
Helpers shared by these scripts, `litdd/evaluation/run_bert_benchmark.py` and
`litdd/experiments/` live in `screen_common.py`.

## Release path

The released screen, [`tmy100000001/LitDD_BERT`](https://huggingface.co/tmy100000001/LitDD_BERT),
was trained by `finetune_seeds.py` on the training set described in the manuscript's Methods
(the annotated set with the confirmed positives, the external curated positives and the corpus
negatives). Every seed is saved under `<save_dir>/seed_<n>`; the released checkpoint is seed 44.

| # | Script | Reads | Writes |
|---|---|---|---|
| 1 | `final_traintest_dataset.py` | `data/annotated_pmid.csv` | `ds_bert_train`, `ds_test` (group-level split, `--group_col {tiab,pmid,gene,g2p_id}`) |
| 2 | `merge_screen_annotations.py` | annotated CSV, annotation worksheet, G2P CSV | the annotated set with the confirmed worksheet rows, collapsed per PMID |
| 3 | `finetune_external_recall.py` | `ds_bert_train`, `ds_test`, worksheet, external truth CSV, random sample | per-variant metrics and per-paper scores for the base set plus external positives |
| 4 | `build_corpus_negatives.py` | converted PubMed shards, G2P snapshots, truth and exclusion CSVs | decade-stratified corpus negatives CSV |
| 5 | `build_prevalence_ladder.py` | a training dataset, the corpus negatives CSV | one dataset per `--add` count under `<out_root>/add<n>` |
| 6 | `cv_hp_search_bert.py` | a training dataset | `--out_json` with fold F1 per grid point and the selected hyperparameters |
| 7 | `finetune_seeds.py` | a training dataset, `ds_test`, external truth CSV, random sample | per-seed metrics CSV and, with `--save_dir`, every seed's checkpoint |

`build_heldout_splits.py` builds the training sets of the stricter held-out evaluations: it
removes from the released training set every row of a group held out by the split under
evaluation (a gene or G2P entry from `final_traintest_dataset.py --group_col`, or papers
published after a cutoff year), after which steps 5 and 7 are re-run on the filtered set.

`bert_finetune.py` trains a single model on a training dataset with the selected
hyperparameters and evaluates it once on `ds_test`; it is the trainer used for the baseline
comparison in `litdd/evaluation/run_bert_benchmark.py` and did not produce the released
checkpoint.

## Not here

- `litdd/experiments/` — ablations that are not on the release path (see its README).
- `data/annotated_pmid.csv` — the annotation input, kept out of the code directories.
