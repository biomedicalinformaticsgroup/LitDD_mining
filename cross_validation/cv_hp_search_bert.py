#!/usr/bin/env python3
"""5-fold StratifiedGroupKFold hyperparameter search for the BERT classifier.

Operates *only on the training set* — the held-out test set is never loaded
here. For each (learning_rate, weight_decay, ...) combination in the grid,
trains a fresh BERT for ``--epochs`` epochs on 4 folds and scores fold-F1 on
the 5th. The combination with the highest mean fold F1 is written to
``--out_json`` (consumed downstream by ``train_test/bert_finetune.py
--hp_json …``).

The grouping column (``--group_col``, default ``tiab``) prevents the same
abstract appearing in both the training and validation halves of any fold.

Default grid is intentionally small (``lr × weight_decay`` = 4 combos × 5
folds = 20 trainings) so a sweep fits in a few hours on 1× A100. Override
via ``--lr_grid`` / ``--wd_grid`` / ``--epochs_grid``.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import time
from typing import Iterable

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evaluate
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def parse_floats(s: Iterable[str]) -> list[float]:
    return [float(x) for x in s]


def parse_ints(s: Iterable[str]) -> list[int]:
    return [int(x) for x in s]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default="train_test/ds_bert_train")
    p.add_argument("--input_model", default="answerdotai/ModernBERT-large")
    p.add_argument("--group_col", default="tiab",
                   help="Column to group on for StratifiedGroupKFold (default 'tiab').")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="cross_validation/bert_hp_search.json")

    # Grid (each flag accepts multiple values)
    p.add_argument("--lr_grid", nargs="+", default=["1e-5", "3e-5"],
                   help="Learning-rate grid.")
    p.add_argument("--wd_grid", nargs="+", default=["0.1", "0.3"],
                   help="Weight-decay grid.")
    p.add_argument("--epochs_grid", nargs="+", default=["3"],
                   help="Epochs grid (single value typical).")

    # Fixed (not searched in default config)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=64)
    return p.parse_args()


def make_compute_metrics():
    f1 = evaluate.load("f1")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")

    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "eval_f1": f1.compute(predictions=preds, references=labels)["f1"],
            "eval_precision": prec.compute(predictions=preds, references=labels, zero_division=0)["precision"],
            "eval_recall": rec.compute(predictions=preds, references=labels, zero_division=0)["recall"],
        }
    return fn


def evaluate_one_fold(
    ds_train_full: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    input_model: str,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    train_bs: int,
    eval_bs: int,
    seed: int,
    fold_idx: int,
) -> dict:
    print(f"\n  -- fold {fold_idx} --", flush=True)
    fold_train = ds_train_full.select(train_idx.tolist())
    fold_val = ds_train_full.select(val_idx.tolist())

    tokenizer = AutoTokenizer.from_pretrained(input_model)

    def preprocess(b):
        return tokenizer(b["tiab"], truncation=True, max_length=512)  # noqa: F821 (closure over tokenizer)

    keep = {"tiab", "label"}

    def tok(ds):
        return ds.map(preprocess, batched=True,
                      remove_columns=[c for c in ds.column_names if c not in keep])

    tok_train = tok(fold_train)
    tok_val = tok(fold_val)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    model = AutoModelForSequenceClassification.from_pretrained(input_model, num_labels=2)
    targs = TrainingArguments(
        output_dir=f"./_cvfold_{fold_idx}",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="no",
        save_strategy="no",
        seed=seed,
        report_to=[],
        logging_steps=200,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tok_train,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )
    t0 = time.time()
    trainer.train()
    metrics = trainer.evaluate(tok_val)
    runtime = time.time() - t0
    metrics["runtime_s"] = round(runtime, 1)

    del trainer, model, tokenizer, tok_train, tok_val
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "fold": fold_idx,
        "f1": float(metrics["eval_f1"]),
        "precision": float(metrics["eval_precision"]),
        "recall": float(metrics["eval_recall"]),
        "runtime_s": metrics["runtime_s"],
    }


def main() -> int:
    args = parse_args()
    ds_train = load_from_disk(args.train_ds_dir)

    if args.group_col not in ds_train.column_names:
        raise SystemExit(f"--group_col '{args.group_col}' not in dataset columns "
                         f"{ds_train.column_names}; pass e.g. --group_col tiab.")

    y = np.array(ds_train["label"], dtype=int)
    groups = np.array(ds_train[args.group_col], dtype=object)
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_assignments = list(sgkf.split(np.zeros(len(ds_train)), y, groups))

    lrs = parse_floats(args.lr_grid)
    wds = parse_floats(args.wd_grid)
    epochs_list = parse_ints(args.epochs_grid)

    grid = list(itertools.product(lrs, wds, epochs_list))
    print(f"[Info] HP grid: {len(grid)} combos × {args.n_folds} folds = "
          f"{len(grid) * args.n_folds} BERT trainings.")
    print(f"[Info] groups: train_size={len(ds_train)} unique_{args.group_col}={len(set(groups))}")

    results = []
    for combo_idx, (lr, wd, epochs) in enumerate(grid, start=1):
        print(f"\n=== combo {combo_idx}/{len(grid)} : lr={lr} wd={wd} epochs={epochs} ===")
        fold_metrics = []
        for fold_idx, (tr_idx, va_idx) in enumerate(fold_assignments, start=1):
            fm = evaluate_one_fold(
                ds_train_full=ds_train,
                train_idx=tr_idx,
                val_idx=va_idx,
                input_model=args.input_model,
                learning_rate=lr,
                weight_decay=wd,
                epochs=epochs,
                train_bs=args.train_bs,
                eval_bs=args.eval_bs,
                seed=args.seed,
                fold_idx=fold_idx,
            )
            print(f"    fold {fold_idx}: f1={fm['f1']:.4f} prec={fm['precision']:.4f} "
                  f"rec={fm['recall']:.4f} ({fm['runtime_s']:.0f}s)")
            fold_metrics.append(fm)
        f1s = np.array([m["f1"] for m in fold_metrics])
        results.append({
            "learning_rate": lr,
            "weight_decay": wd,
            "epochs": epochs,
            "train_bs": args.train_bs,
            "fold_f1": list(f1s),
            "mean_f1": float(f1s.mean()),
            "std_f1": float(f1s.std(ddof=1)),
        })
        print(f"  mean F1 = {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f}")

    results.sort(key=lambda r: -r["mean_f1"])
    best = results[0]
    out = {
        "n_folds": args.n_folds,
        "input_model": args.input_model,
        "group_col": args.group_col,
        "results": results,
        "best": {k: best[k] for k in ("learning_rate", "weight_decay", "epochs", "train_bs")},
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Info] best HPs (mean F1 {best['mean_f1']:.4f}): {out['best']}")
    print(f"[Info] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
