#!/usr/bin/env python3
"""Benchmark baseline checkpoints against the screen under one protocol.

Reads a training dataset and the ``ds_test`` split (``--train_ds_dir`` and ``--test_ds_dir``,
or ``<data_dir>/ds_bert_train`` and ``<data_dir>/ds_test``). For every baseline the same three
steps are applied: hyperparameter selection by stratified group k-fold CV on the training set
(``--cv_hp_search`` runs ``cv_hp_search_bert`` per baseline; otherwise ``--hp_json`` or the
flag defaults are used), a refit on the full training set, and one evaluation on the test set.
``--litdd_model_path`` scores an already fine-tuned checkpoint without training.

Writes one row per (model, seed) to ``--out_csv``: precision, recall, F1 and the confusion
counts on the test set, plus recall on ``--external_csv`` (per source and overall) when given.
``--pred_dir`` receives the per-example test predictions and external probabilities, which
paired tests such as McNemar's need. ``--skip_existing`` resumes an interrupted sweep.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from litdd.training.screen_common import (
    append_csv_row,
    gene_fold,
    load_existing,
    make_compute_metrics,
    maybe_load_hp_json,
    score_proba,
)

DEFAULT_BASELINES = [
    "answerdotai/ModernBERT-large",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-v1.1",
]
MAX_MODEL_LENGTH = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default=None,
                   help="Training dataset; overrides <data_dir>/ds_bert_train.")
    p.add_argument("--test_ds_dir", default=None, help="Test dataset; overrides <data_dir>/ds_test.")
    p.add_argument("--data_dir", default="data",
                   help="Directory containing ds_bert_train and ds_test.")
    p.add_argument("--out_csv", default="results/bert_results.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override baseline list.")
    p.add_argument("--litdd_label", default=None,
                   help="Row label for --litdd_model_path (default: the path itself).")
    p.add_argument("--litdd_model_path", default=None,
                   help="If set, evaluate a fine-tuned checkpoint without training and add a row.")
    p.add_argument("--hp_json", default=None,
                   help="JSON of selected HPs to use for every baseline (output of "
                        "cv_hp_search_bert). If omitted, the --learning_rate / --epochs / "
                        "--weight_decay defaults apply.")
    p.add_argument("--cv_hp_search", action="store_true",
                   help="Run a CV HP search per baseline (runs cv_hp_search_bert in a sub-process).")
    p.add_argument("--cv_search_module", default="litdd.training.cv_hp_search_bert",
                   help="Module run with `python -m` for the per-baseline CV search.")

    # Hyperparameters used when neither --hp_json nor --cv_hp_search is set.
    p.add_argument("--learning_rate", type=float, default=1.736e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip (model, seed) pairs already present in --out_csv.")
    p.add_argument("--skip_baselines", action="store_true",
                   help="Evaluate only --litdd_model_path; no baseline is trained.")
    p.add_argument("--external_csv", default=None,
                   help="External truth CSV (pmid, tiab, source[, gene]) scored for recall by every model.")
    p.add_argument("--external_scope", choices=["raw", "heldout_gene_fold"], default="raw",
                   help="'raw' scores every row; 'heldout_gene_fold' keeps the papers whose genes "
                        "all fall in fold 0 of 10.")
    p.add_argument("--external_threshold", type=float, default=0.5)
    p.add_argument("--pred_dir", default=None,
                   help="Directory for per-example test predictions and external probabilities, "
                        "one CSV per (model, seed).")
    return p.parse_args()


def tokenize(ds, tokenizer, keep: set[str] | None = None):
    """Tokenise ``tiab`` up to the model's own length (capped at 8,192) and drop the other columns."""
    keep = {"tiab", "label"} if keep is None else keep
    limit = getattr(tokenizer, "model_max_length", 512) or 512
    limit = min(limit, MAX_MODEL_LENGTH)  # some tokenizers report a sentinel such as 1e30

    def fn(b):
        return tokenizer(b["tiab"], truncation=True, max_length=limit)
    return ds.map(fn, batched=True,
                  remove_columns=[c for c in ds.column_names if c not in keep])


def dump_predictions(trainer, tok_test, ds_test, pred_dir: str, model_name: str, seed: int) -> None:
    """Write ``idx, label, pred`` for every test example to ``<pred_dir>/<model>__seed<seed>.csv``."""
    os.makedirs(pred_dir, exist_ok=True)
    logits = trainer.predict(tok_test).predictions
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(ds_test["label"])
    safe = model_name.replace("/", "__")
    path = os.path.join(pred_dir, f"{safe}__seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pred"])
        for i, (lab, p) in enumerate(zip(labels, preds)):
            w.writerow([i, int(lab), int(p)])
    print(f"[INFO] wrote {len(preds)} predictions -> {path}", flush=True)


def hp_search_for_model(model_name: str, args: argparse.Namespace, train_path: str) -> dict:
    """Run the CV search module for one baseline on ``train_path`` and return its ``best`` entry."""
    out_json = f"_hp_search_{model_name.replace('/', '__')}.json"
    cmd = [
        sys.executable, "-m", args.cv_search_module,
        "--train_ds_dir", train_path,
        "--input_model", model_name,
        "--out_json", out_json,
        "--seed", str(args.seed),
    ]
    print(f"\n[CV] HP search for {model_name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        return json.load(f)["best"]


def _metrics_row(model: str, seed: int, metrics: dict) -> dict:
    return {
        "model": model, "seed": seed,
        "precision": round(float(metrics["eval_precision"]), 6),
        "recall": round(float(metrics["eval_recall"]), 6),
        "f1": round(float(metrics["eval_f1"]), 6),
        "tp": int(metrics["eval_tp"]), "fp": int(metrics["eval_fp"]),
        "fn": int(metrics["eval_fn"]), "tn": int(metrics["eval_tn"]),
    }


def fine_tune_and_eval(model_name: str, hp: dict, args: argparse.Namespace, ds_train, ds_test) -> dict:
    """Fine-tune ``model_name`` with ``hp`` on ``ds_train`` in fp32 and evaluate once on ``ds_test``."""
    from transformers import set_seed

    # Seed before from_pretrained so the classification-head initialisation follows the seed.
    set_seed(args.seed)
    print(f"\n=== Refit + test: {model_name} (HPs: {hp}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    tok_train = tokenize(ds_train, tokenizer)
    tok_test = tokenize(ds_test, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=f"./_bench_{model_name.replace('/', '__')}",
        learning_rate=hp.get("learning_rate", args.learning_rate),
        per_device_train_batch_size=int(hp.get("train_bs", args.train_bs)),
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=int(hp.get("epochs", args.epochs)),
        weight_decay=hp.get("weight_decay", args.weight_decay),
        eval_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(accuracy=True, counts=True),
    )
    trainer.train()
    row = _metrics_row(model_name, args.seed, trainer.evaluate(tok_test))
    if args.pred_dir:
        dump_predictions(trainer, tok_test, ds_test, args.pred_dir, model_name, args.seed)
    if args.external_csv:
        row.update(external_recall(
            model, tokenizer, args.external_csv, args.external_scope,
            args.external_threshold,
            pred_path=(os.path.join(args.pred_dir,
                                    f"EXT__{model_name.replace('/', '__')}__seed{args.seed}.csv")
                       if args.pred_dir else None)))
    return row


def external_recall(model, tokenizer, external_csv: str, scope: str,
                    threshold: float, max_length: int | None = None,
                    pred_path: str | None = None) -> dict:
    """Recall on an external truth corpus, overall and per ``source``.

    ``scope == "heldout_gene_fold"`` keeps the papers whose genes all fall in fold 0 of 10.
    Texts are truncated to the model's position-embedding limit. ``pred_path`` receives
    ``pmid, prob, pred`` per paper.
    """
    limit = getattr(model.config, "max_position_embeddings", 512) or 512
    limit = min(limit, getattr(tokenizer, "model_max_length", limit) or limit)
    max_length = limit if max_length is None else min(max_length, limit)

    ext = pd.read_csv(external_csv, dtype=str).drop_duplicates("pmid")
    if scope == "heldout_gene_fold":
        if "gene" not in ext.columns:
            raise SystemExit("--external_scope heldout_gene_fold needs a 'gene' column")
        folds = ext.groupby("pmid")["gene"].apply(lambda gs: {gene_fold(g, 10) for g in gs})
        ext = ext[ext["pmid"].map(folds).map(lambda f: f == {0})].reset_index(drop=True)
    texts = ext["tiab"].fillna("").tolist()
    probs = score_proba(model, tokenizer, texts, max_length)

    out = {"external_scope": scope, "external_n": len(ext),
           "external_recall_all": round(float((probs >= threshold).mean()), 4)}
    if pred_path:
        os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
        with open(pred_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pmid", "prob", "pred"])
            for pmid, pr in zip(ext["pmid"].tolist(), probs):
                w.writerow([pmid, round(float(pr), 6), int(pr >= threshold)])
        print(f"[INFO] wrote {len(probs)} external predictions -> {pred_path}", flush=True)
    if "source" in ext.columns:
        for src, m in ext.groupby("source").groups.items():
            idx = ext.index.get_indexer(m)
            out[f"external_recall_{src}"] = round(float((probs[idx] >= threshold).mean()), 4)
    return out


def evaluate_only(model_name: str, label: str, ds_test, external_csv: str | None = None,
                  external_scope: str = "raw", external_threshold: float = 0.5, seed: int = 42,
                  pred_dir: str | None = None) -> dict:
    """Score a checkpoint on ``ds_test`` without training.

    A base model gets a freshly initialised classification head, so ``seed`` fixes that
    initialisation and makes the row reproducible.
    """
    from transformers import set_seed

    set_seed(seed)
    print(f"\n=== Eval only: {label} ({model_name}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tok_test = tokenize(ds_test, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(accuracy=True, counts=True),
    )
    row = _metrics_row(label, seed, trainer.evaluate(tok_test))
    if external_csv:
        row.update(external_recall(
            model, tokenizer, external_csv, external_scope, external_threshold,
            pred_path=(os.path.join(pred_dir, f"EXT__{label.replace('/', '__')}.csv")
                       if pred_dir else None)))
    return row


def main() -> int:
    args = parse_args()
    train_path = args.train_ds_dir or os.path.join(args.data_dir, "ds_bert_train")
    test_path = args.test_ds_dir or os.path.join(args.data_dir, "ds_test")
    # The training set is read only when a baseline is trained.
    ds_train = None if args.skip_baselines else load_from_disk(train_path)
    ds_test = load_from_disk(test_path)
    print(f"train: {train_path}\ntest : {test_path}", flush=True)

    existing = load_existing(args.out_csv) if args.skip_existing else set()
    baselines = [] if args.skip_baselines else (args.models or DEFAULT_BASELINES)
    shared = maybe_load_hp_json(args.hp_json) if args.hp_json else None

    if args.litdd_model_path:
        # The row is named after the checkpoint so several checkpoints can share one CSV.
        label = args.litdd_label or f"eval-only: {args.litdd_model_path}"
        if (label, str(args.seed)) not in existing:
            row = evaluate_only(args.litdd_model_path, label, ds_test,
                                external_csv=args.external_csv,
                                external_scope=args.external_scope,
                                external_threshold=args.external_threshold,
                                seed=args.seed, pred_dir=args.pred_dir)
            append_csv_row(args.out_csv, row)
            print("->", row)

    for name in baselines:
        if (name, str(args.seed)) in existing:
            print(f"[SKIP] {name} seed={args.seed} already in {args.out_csv}")
            continue
        try:
            if args.cv_hp_search:
                hp = hp_search_for_model(name, args, train_path)
            elif shared is not None:
                hp = shared
            else:
                hp = {}  # flag defaults
            row = fine_tune_and_eval(name, hp, args, ds_train, ds_test)
            append_csv_row(args.out_csv, row)
            print("->", row)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            append_csv_row(args.out_csv, {"model": name, "seed": args.seed,
                                          "precision": "", "recall": "", "f1": ""})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
