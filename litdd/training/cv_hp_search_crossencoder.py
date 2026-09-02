#!/usr/bin/env python3
"""5-fold StratifiedGroupKFold hyperparameter search for the cross-encoder.

Operates only on the training set. For each (learning_rate, epochs) combination
in the grid, fine-tunes a fresh cross-encoder on 4 folds of the labeled-pair
dataset and scores the 5th. The combination with the highest mean fold F1 is
written to ``--out_json`` (consumed by ``crossencode_finetune.py --hp_json``);
per-fold rows go to ``--out_csv``.

Leakage / rigour properties, each enforced here rather than assumed:

* **Grouped folds** — ``StratifiedGroupKFold`` on ``--group_col`` (default
  ``tiab``); fold disjointness is asserted at runtime.
* **Hard negatives are mined inside each fold's train half**, from that fold's
  positives only, so a validation abstract never shapes the training pairs.
  Mining depends on the fold, not the HP combo, so each fold is mined once and
  the result reused across the whole grid.
* **Annotated negatives included** — the human-labeled negative pairs from the
  fold's train half join the mined negatives (they are gene-sharing near-misses,
  the candidate distribution deployment actually scores). ``--no_annotated_negatives``
  reproduces the original mined-only recipe.
* **Seed set before model construction** (the transformers trainer only seeds
  after the model is loaded).
* **Validation metrics at a fixed 0.5 threshold** from raw ``model.predict``
  scores in deployment pair order ``(tiab, thread)`` — no threshold tuned on the
  validation half, and no reliance on evaluator classes that do so.

Default grid is small (``lr x epochs`` = 2 combos x 5 folds = 10 trainings).
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold


def parse_floats(s: Iterable[str]) -> list[float]:
    return [float(x) for x in s]


def parse_ints(s: Iterable[str]) -> list[int]:
    return [int(x) for x in s]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default="litdd/training/ds_cross_train",
                   help="Pair-level training dataset (tiab, g2p_lgmde, label).")
    p.add_argument("--corpus_json", default=None,
                   help="corpus.json from build_crossencoder_dataset.py — the mining "
                        "candidate pool in this arm's rendering.")
    p.add_argument("--g2p_corpus_csv", default=None,
                   help="Raw G2P export; flat-rendered as the corpus if no --corpus_json.")
    p.add_argument("--input_model", default="ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--embed_model", default="abhinand/MedEmbed-large-v0.1",
                   help="Sentence-transformer used by mine_hard_negatives.")
    p.add_argument("--group_col", default="tiab")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--no_annotated_negatives", dest="include_annotated_negatives",
                   action="store_false")
    p.add_argument("--tag", default="",
                   help="Free-text label (e.g. the thread-variant arm) recorded per CSV row.")
    p.add_argument("--out_json", default="litdd/training/crossencoder_hp_search.json")
    p.add_argument("--out_csv", default=None,
                   help="Per-fold, per-HP rows; appended to, so several arms can share one file.")

    # Grid
    p.add_argument("--lr_grid", nargs="+", default=["1e-5", "3e-5"])
    p.add_argument("--epochs_grid", nargs="+", default=["2"])
    # Fixed
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    return p.parse_args()


def load_corpus(corpus_json: str | None, g2p_csv: str | None) -> list[str]:
    if corpus_json:
        with open(corpus_json) as f:
            return sorted(set(str(v) for v in json.load(f).values()))
    if g2p_csv:
        from litdd.threads import build_lgmde_list
        return build_lgmde_list(g2p_csv)
    raise SystemExit("one of --corpus_json / --g2p_corpus_csv is required")


def mine_all_folds(df: pd.DataFrame, fold_assignments: list, args,
                   corpus: list[str]) -> list[Dataset]:
    """Labeled-pair training set for each fold, mined from that fold's train half only.

    Mining is a function of (fold, corpus, embedder) — not of the HP combo — so it runs
    once per fold, before the grid loop, with a single embedder load.
    """
    from sentence_transformers import SentenceTransformer

    from litdd.training.mine_hard_negatives import mine_labeled_pairs

    embedder = SentenceTransformer(args.embed_model)
    fold_train_sets: list[Dataset] = []
    for fold_idx, (tr_idx, _) in enumerate(fold_assignments, start=1):
        fold_train = df.iloc[tr_idx]
        print(f"[Info] mining fold {fold_idx}: {int((fold_train['label'] == 1).sum())} "
              f"positives", flush=True)
        ds = Dataset.from_pandas(fold_train[["tiab", "g2p_lgmde", "label"]],
                                 preserve_index=False)
        fold_train_sets.append(mine_labeled_pairs(
            ds, corpus, args.embed_model,
            include_annotated_negatives=args.include_annotated_negatives,
            embedder=embedder,
        ))
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fold_train_sets


def eval_pairs(model, df_val: pd.DataFrame, batch_size: int) -> dict:
    """Fixed-threshold (0.5) metrics from raw scores, in deployment pair order."""
    pairs = list(zip(df_val["tiab"], df_val["g2p_lgmde"]))
    scores = np.asarray(model.predict(pairs, batch_size=batch_size)).reshape(-1)
    labels = df_val["label"].astype(int).to_numpy()
    preds = (scores >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    ap = float(average_precision_score(labels, scores)) if labels.any() else 0.0
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec), "ap": ap}


def train_one_fold(train_ds: Dataset, df_val: pd.DataFrame, args,
                   learning_rate: float, epochs: int, fold_idx: int) -> dict:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import set_seed

    print(f"\n  -- fold {fold_idx} --", flush=True)
    set_seed(args.seed)  # BEFORE model construction, so any head init is governed by it
    model = CrossEncoder(args.input_model)
    loss = BinaryCrossEntropyLoss(model)

    targs = CrossEncoderTrainingArguments(
        output_dir=f"./_cvce_fold_{fold_idx}",
        num_train_epochs=epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="no",
        save_strategy="no",
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        report_to=[],
    )
    trainer = CrossEncoderTrainer(model=model, args=targs, train_dataset=train_ds,
                                  loss=loss)
    t0 = time.time()
    trainer.train()
    runtime = time.time() - t0

    metrics = eval_pairs(model, df_val, args.eval_bs)
    metrics.update({"fold": fold_idx, "runtime_s": round(runtime, 1)})

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def append_csv(path: str, row: dict) -> None:
    fieldnames = ["tag", "learning_rate", "epochs", "train_bs", "warmup_ratio",
                  "precision_mode", "fold", "f1", "precision", "recall", "ap",
                  "runtime_s"]
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> int:
    args = parse_args()
    ds_train = load_from_disk(args.train_ds_dir)
    if args.group_col not in ds_train.column_names:
        raise SystemExit(f"--group_col '{args.group_col}' not in dataset columns "
                         f"{ds_train.column_names}.")
    df = ds_train.to_pandas()
    corpus = load_corpus(args.corpus_json, args.g2p_corpus_csv)
    print(f"[Info] {len(df)} pairs, {len(corpus)} corpus threads, tag={args.tag!r}")

    y = df["label"].astype(int).values
    groups = df[args.group_col].values
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_assignments = list(sgkf.split(np.zeros(len(df)), y, groups))

    # Grouping is the leakage control: verify it rather than trust it.
    for fold_idx, (tr_idx, va_idx) in enumerate(fold_assignments, start=1):
        shared = set(groups[tr_idx]) & set(groups[va_idx])
        assert not shared, f"fold {fold_idx}: {len(shared)} groups on both sides"
    print(f"[Info] fold grouping verified: no '{args.group_col}' value crosses a fold")

    fold_train_sets = mine_all_folds(df, fold_assignments, args, corpus)

    lrs = parse_floats(args.lr_grid)
    epochs_list = parse_ints(args.epochs_grid)
    grid = list(itertools.product(lrs, epochs_list))
    print(f"[Info] HP grid: {len(grid)} combos x {args.n_folds} folds = "
          f"{len(grid) * args.n_folds} cross-encoder trainings.")

    results = []
    for combo_idx, (lr, epochs) in enumerate(grid, start=1):
        print(f"\n=== combo {combo_idx}/{len(grid)} : lr={lr} epochs={epochs} ===")
        fold_metrics = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_assignments, start=1):
            fm = train_one_fold(fold_train_sets[fold_idx - 1], df.iloc[va_idx], args,
                                learning_rate=lr, epochs=epochs, fold_idx=fold_idx)
            print(f"    fold {fold_idx}: f1={fm['f1']:.4f} ap={fm['ap']:.4f} "
                  f"({fm['runtime_s']:.0f}s)")
            fold_metrics.append(fm)
            if args.out_csv:
                append_csv(args.out_csv, {
                    "tag": args.tag, "learning_rate": lr, "epochs": epochs,
                    "train_bs": args.train_bs, "warmup_ratio": args.warmup_ratio,
                    "precision_mode": args.precision, **fm})
        f1s = np.array([m["f1"] for m in fold_metrics])
        results.append({
            "learning_rate": lr,
            "epochs": epochs,
            "train_bs": args.train_bs,
            "warmup_ratio": args.warmup_ratio,
            "fold_f1": list(f1s),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
        })
        print(f"  mean F1 = {f1s.mean():.4f} ± {results[-1]['std_f1']:.4f}")

    results.sort(key=lambda r: -r["mean_f1"])
    best = results[0]
    out = {
        "tag": args.tag,
        "n_folds": args.n_folds,
        "input_model": args.input_model,
        "group_col": args.group_col,
        "precision": args.precision,
        "include_annotated_negatives": args.include_annotated_negatives,
        "results": results,
        "best": {k: best[k] for k in ("learning_rate", "epochs", "train_bs", "warmup_ratio")},
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Info] best HPs (mean F1 {best['mean_f1']:.4f}): {out['best']}")
    print(f"[Info] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
