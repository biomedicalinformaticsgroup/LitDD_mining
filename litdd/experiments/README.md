# Experiments

Ablations of the screen that are not on the release path; none of them produced the released
models (for those see `litdd/training/README.md`). They import their training and scoring
helpers from `litdd/training/screen_common.py` and are run as modules from the repository
root, for example `python -m litdd.experiments.finetune_screen_2x2 --help`. Output paths are
required flags.

| Script | Question answered |
|---|---|
| `finetune_screen_2x2.py` | does augmentation, gene-conditioning of the input, or both, change test F1 and the recovery of gene-present screen misses |
| `finetune_augmented_screen.py` | does adding the confirmed augmentation positives change test F1, miss recovery and the positive rate on random PubMed; learning curve over the number of positives; per-example scores for threshold sweeps |
| `finetune_external_curve.py` | does held-out external recall keep rising as more gene buckets of external positives enter training |
| `build_gene_conditioned_dataset.py` | builds the (abstract, candidate gene) training set the gene-conditioned variants read |
