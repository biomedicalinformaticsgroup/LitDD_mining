# Training

Which script produced what. The distinction matters for reproducibility (Reviewer 2 R2-R1 /
R2-S4): the released models come from a specific subset of these, and the revision-era
ablations live in `experiments/` rather than here.

## Canonical path — reproduces the released models

Run in this order (or via `../../run_pipeline.sh --full`):

| # | Script | Produces |
|---|---|---|
| 1 | `final_traintest_dataset.py` | group-level train/test split (`--group_col {pmid,tiab,gene,g2p_id}`) |
| 2 | `merge_screen_annotations.py` | the augmented annotation set, merging the original labels with the reviewer-confirmed molecular-framed positives |
| 3 | `cv_hp_search_bert.py` | screen hyperparameters — 5-fold StratifiedGroupKFold on the **training portion only** |
| 4 | `finetune_seeds.py` | **the released screen** — [`tmy100000001/LitDD_BERT`](https://huggingface.co/tmy100000001/LitDD_BERT) |
| 5 | `mine_hard_negatives.py` | hard negatives for the cross-encoder (`abhinand/MedEmbed-large-v0.1`, 5 per positive, rank 5–50) |
| 6 | `cv_hp_search_crossencoder.py` | cross-encoder hyperparameters, hard negatives re-mined per fold |
| 7 | `crossencode_finetune.py` | **the released cross-encoder** — [`tmy100000001/LitDD_crossencoder`](https://huggingface.co/tmy100000001/LitDD_crossencoder) |

`bert_finetune.py` is the plain screen trainer used for the fair-baseline comparison (Table 1)
and for the held-out/time-split evaluations. It is *not* what produced the released checkpoint.

**The released screen came from `finetune_seeds.py`**, not from `bert_finetune.py`. The name is
historical — it seed-averages *and* saves the checkpoint. Exact invocation:
`revision/litdd_lock_job.yaml` (fixed seed 42, CV-selected lr 3e-5 / wd 0.1 / 5 epochs).

`finetune_external_recall.py` built the high-recall training set (`ds_hirecall_train`) that
step 4 consumes, so it is on the release path even though it began as a revision experiment.

## Not here

- `experiments/` — revision-era ablations: the 2×2 factorial, learning curves, the FPR proxy,
  and gene-conditioned dataset construction. None feed the released models.
- `data/annotated_pmid.csv` — the annotation input, kept out of the code directories.
