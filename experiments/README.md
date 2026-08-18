# Revision experiments

One-off analyses run to answer specific reviewer points. **None of these produced the released
models** — for that see `litdd/training/README.md`. They are kept because the responses cite
their results, and because a reviewer should be able to re-run them.

| Script | Reviewer point | What it answers |
|---|---|---|
| `finetune_screen_2x2.py` | R3.4 / R1.3 | 2×2 factorial: augmentation × gene-conditioning |
| `finetune_augmented_screen.py` | R2-P1 / R3.4 | augmented-screen fine-tune; deployment-FPR proxy and precision projection (`--random_csv`); recall/precision and threshold sweeps |
| `finetune_external_curve.py` | R3.3 | learning curve over external training data; also saved the `screen_hirecall_v1/v2` checkpoints |
| `build_gene_conditioned_dataset.py` | R3.4 / R1.3 | gene-conditioned dataset variant for the screen |

Their k8s job manifests are in `revision/` (gitignored — local working area).
