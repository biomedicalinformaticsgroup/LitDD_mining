#!/usr/bin/env python3
"""5-fold StratifiedGroupKFold hyperparameter search for the cross-encoder.

Operates only on the training set. For each (learning_rate, epochs, ...)
combination in the grid, fine-tunes a fresh cross-encoder on 4 folds of
the (anchor=tiab, positive=g2p_lgmde) hard-negatives dataset and scores
fold AUC / binary-F1 on the 5th. The combination with the highest mean
fold F1 is written to ``--out_json`` (consumed by
``train_test/crossencode_finetune.py --hp_json …``).

Hard negatives are mined *inside each fold's train half* so a positive's
nearest negatives never come from the validation half (no leakage).

Default grid is small (``lr × epochs`` = 2 combos × 5 folds = 10
trainings).
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.util import mine_hard_negatives
from sklearn.model_selection import StratifiedGroupKFold


def parse_floats(s: Iterable[str]) -> list[float]:
    return [float(x) for x in s]


def parse_ints(s: Iterable[str]) -> list[int]:
    return [int(x) for x in s]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default="train_test/ds_cross_train",
                   help="Pair-level training dataset (tiab, g2p_lgmde, label).")
    p.add_argument("--g2p_corpus_csv", required=True,
                   help="Full G2P corpus CSV (used as candidate negatives during mining).")
    p.add_argument("--input_model", default="ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--embed_model", default="abhinand/MedEmbed-large-v0.1",
                   help="Sentence-transformer used by mine_hard_negatives.")
    p.add_argument("--group_col", default="tiab")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="cross_validation/crossencoder_hp_search.json")

    # Grid
    p.add_argument("--lr_grid", nargs="+", default=["1e-5", "3e-5"])
    p.add_argument("--epochs_grid", nargs="+", default=["2"])
    # Fixed
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    return p.parse_args()


def mine_negatives(fold_pos_df: pd.DataFrame, embed_model: str,
                   g2p_corpus: list[str]) -> Dataset:
    """Mine hard negatives inside the training half of one fold."""
    ds = Dataset.from_pandas(fold_pos_df[["tiab", "g2p_lgmde"]], preserve_index=False)
    embedder = SentenceTransformer(embed_model)
    out = mine_hard_negatives(
        dataset=ds,
        model=embedder,
        anchor_column_name="tiab",
        positive_column_name="g2p_lgmde",
        corpus=g2p_corpus,
        range_min=5,
        range_max=50,
        max_score=0.95,
        relative_margin=0.01,
        num_negatives=5,
        sampling_strategy="top",
        batch_size=128,
        output_format="labeled-pair",
        use_faiss=False,
    )
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def evaluate_one_fold(
    fold_train: pd.DataFrame,
    fold_val: pd.DataFrame,
    g2p_corpus: list[str],
    args,
    learning_rate: float,
    epochs: int,
    fold_idx: int,
) -> dict:
    print(f"\n  -- fold {fold_idx} --", flush=True)
    fold_train_pos = fold_train[fold_train["label"] == 1]
    if fold_train_pos.empty:
        return {"fold": fold_idx, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "ap": 0.0, "runtime_s": 0.0}
    hard_negs_ds = mine_negatives(fold_train_pos, args.embed_model, g2p_corpus)

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
        report_to=[],
    )
    trainer = CrossEncoderTrainer(
        model=model, args=targs, train_dataset=hard_negs_ds, loss=loss,
    )
    t0 = time.time()
    trainer.train()
    runtime = time.time() - t0

    val_eval = CrossEncoderClassificationEvaluator(
        sentence_pairs=list(zip(fold_val["g2p_lgmde"], fold_val["tiab"])),
        labels=list(fold_val["label"].astype(int)),
        name=f"cv_fold_{fold_idx}",
    )
    metrics = val_eval(model)
    f1 = float(metrics.get(val_eval.primary_metric, metrics.get("binary_f1", 0.0)))
    prec = float(metrics.get("binary_precision", 0.0))
    rec = float(metrics.get("binary_recall", 0.0))
    ap = float(metrics.get("average_precision", 0.0))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"fold": fold_idx, "f1": f1, "precision": prec, "recall": rec,
            "ap": ap, "runtime_s": round(runtime, 1)}


def main() -> int:
    args = parse_args()
    ds_train = load_from_disk(args.train_ds_dir)
    if args.group_col not in ds_train.column_names:
        raise SystemExit(f"--group_col '{args.group_col}' not in dataset columns "
                         f"{ds_train.column_names}.")
    df = ds_train.to_pandas()

    g2p_csv = pd.read_csv(args.g2p_corpus_csv, dtype=str, keep_default_na=False)
    if "g2p_lgmde" not in g2p_csv.columns:
        # Build the LGMDE join string from the canonical G2P columns
        # (matches train_test/mine_hard_negatives.py).
        lgmde_cols = [
            "g2p id", "gene symbol", "gene mim", "hgnc id", "previous gene symbols",
            "disease name", "disease mim", "disease MONDO", "allelic requirement",
            "cross cutting modifier", "confidence", "inferred variant consequence",
            "variant types", "molecular mechanism", "molecular mechanism categorisation",
        ]
        present = [c for c in lgmde_cols if c in g2p_csv.columns]
        if not present:
            raise SystemExit(f"{args.g2p_corpus_csv} has neither 'g2p_lgmde' nor any of "
                             f"the canonical G2P columns {lgmde_cols}.")
        g2p_csv["g2p_lgmde"] = g2p_csv[present].astype(str).agg(" - ".join, axis=1)
    g2p_corpus = list(g2p_csv["g2p_lgmde"])

    y = df["label"].astype(int).values
    groups = df[args.group_col].values
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_assignments = list(sgkf.split(np.zeros(len(df)), y, groups))

    lrs = parse_floats(args.lr_grid)
    epochs_list = parse_ints(args.epochs_grid)
    grid = list(itertools.product(lrs, epochs_list))
    print(f"[Info] HP grid: {len(grid)} combos × {args.n_folds} folds = "
          f"{len(grid) * args.n_folds} cross-encoder trainings.")

    results = []
    for combo_idx, (lr, epochs) in enumerate(grid, start=1):
        print(f"\n=== combo {combo_idx}/{len(grid)} : lr={lr} epochs={epochs} ===")
        fold_metrics = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_assignments, start=1):
            fm = evaluate_one_fold(
                fold_train=df.iloc[tr_idx],
                fold_val=df.iloc[va_idx],
                g2p_corpus=g2p_corpus,
                args=args,
                learning_rate=lr,
                epochs=epochs,
                fold_idx=fold_idx,
            )
            print(f"    fold {fold_idx}: f1={fm['f1']:.4f} ap={fm['ap']:.4f} "
                  f"({fm['runtime_s']:.0f}s)")
            fold_metrics.append(fm)
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
        "n_folds": args.n_folds,
        "input_model": args.input_model,
        "group_col": args.group_col,
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
