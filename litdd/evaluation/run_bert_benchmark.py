#!/usr/bin/env python3
"""Fair baseline-vs-LitDD-BERT benchmark.

Methodology, applied identically to LitDD-BERT and every baseline checkpoint:

  1. **Hyperparameter selection** by 5-fold ``StratifiedGroupKFold`` CV on
     the training set only. (Re-uses ``litdd/training/cv_hp_search_bert.py``
     when ``--cv_hp_search`` is passed; otherwise the baseline reuses the
     ``--hp_json`` produced for LitDD-BERT.)
  2. **Refit** on the *full* training set with the selected HPs.
  3. **Evaluate once** on the untouched test set.

The earlier draft of this script loaded each baseline with
``num_labels=2, ignore_mismatched_sizes=True`` and skipped fine-tuning
entirely — i.e. evaluated a randomly-initialised classification head.
Such a head has never seen any training data so it predicts near-randomly
on a 2-class task, which is why baseline F1 looked like ~15% (≈ random
chance). The current script fine-tunes every baseline with the same
hyperparameter-selection protocol as LitDD-BERT, which is the standard for
fair comparison on a binary classification task.

Inputs:  ``ds_bert_train``, ``ds_test`` from ``final_traintest_dataset.py``.
Output:  ``bert_results.csv`` with columns ``model,precision,recall,f1``.

Per baseline a CV sweep + refit costs roughly the same GPU-time as training
LitDD-BERT (≈ several hours per baseline on 1× A100). Use
``--skip_existing`` to resume an interrupted sweep.
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

import evaluate
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

DEFAULT_BASELINES = [
    "answerdotai/ModernBERT-large",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-v1.1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", default="train_test",
                   help="Directory containing ds_bert_train and ds_test.")
    p.add_argument("--out_csv", default="benchmarking/bert_results.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override baseline list.")
    p.add_argument("--litdd_model_path", default=None,
                   help="If set, evaluate a previously fine-tuned LitDD-BERT checkpoint "
                        "(no re-training) and add a row.")
    p.add_argument("--hp_json", default=None,
                   help="JSON of selected HPs to use for every baseline (output of "
                        "litdd/training/cv_hp_search_bert.py). If omitted, falls back to "
                        "the --learning_rate / --epochs / --weight_decay defaults below.")
    p.add_argument("--cv_hp_search", action="store_true",
                   help="Run a fresh CV HP search per baseline (calls "
                        "litdd/training/cv_hp_search_bert.py). Slower but most rigorous.")
    p.add_argument("--cv_search_script", default="litdd/training/cv_hp_search_bert.py")

    # Fallback HPs (used when neither --hp_json nor --cv_hp_search is set)
    p.add_argument("--learning_rate", type=float, default=1.736e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def load_existing(out_csv: str) -> set[str]:
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {row["model"] for row in csv.DictReader(f) if row.get("model")}


def append_row(out_csv: str, row: dict) -> None:
    is_new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "precision", "recall", "f1"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def tokenize(ds, tokenizer, keep={"tiab", "label"}):
    def fn(b):
        return tokenizer(b["tiab"], truncation=True, max_length=512)
    return ds.map(fn, batched=True,
                  remove_columns=[c for c in ds.column_names if c not in keep])


def make_compute_metrics():
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")
    acc = evaluate.load("accuracy")

    def fn(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "eval_accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "eval_precision": prec.compute(predictions=preds, references=labels, zero_division=0)["precision"],
            "eval_recall": rec.compute(predictions=preds, references=labels, zero_division=0)["recall"],
            "eval_f1": f1.compute(predictions=preds, references=labels)["f1"],
        }
    return fn


def hp_search_for_model(model_name: str, args) -> dict:
    """Run cv_hp_search_bert.py for one baseline and return the chosen HPs."""
    out_json = f"_hp_search_{model_name.replace('/', '__')}.json"
    cmd = [
        sys.executable, args.cv_search_script,
        "--train_ds_dir", os.path.join(args.data_dir, "ds_bert_train"),
        "--input_model", model_name,
        "--out_json", out_json,
        "--seed", str(args.seed),
    ]
    print(f"\n[CV] HP search for {model_name}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    with open(out_json) as f:
        return json.load(f)["best"]


def fine_tune_and_eval(model_name: str, hp: dict, args, ds_train, ds_test) -> dict:
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
        report_to=[],
        logging_steps=200,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )
    trainer.train()
    test_metrics = trainer.evaluate(tok_test)

    return {
        "model": model_name,
        "precision": round(float(test_metrics["eval_precision"]), 6),
        "recall": round(float(test_metrics["eval_recall"]), 6),
        "f1": round(float(test_metrics["eval_f1"]), 6),
    }


def evaluate_only(model_name: str, label: str, ds_test) -> dict:
    print(f"\n=== Eval only: {label} ({model_name}) ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tok_test = tokenize(ds_test, tokenizer)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(),
    )
    metrics = trainer.evaluate(tok_test)
    return {
        "model": label,
        "precision": round(float(metrics["eval_precision"]), 6),
        "recall": round(float(metrics["eval_recall"]), 6),
        "f1": round(float(metrics["eval_f1"]), 6),
    }


def shared_hps(args) -> dict | None:
    if args.hp_json:
        with open(args.hp_json) as f:
            data = json.load(f)
        return data.get("best", data)
    return None


def main() -> int:
    args = parse_args()
    ds_train = load_from_disk(os.path.join(args.data_dir, "ds_bert_train"))
    ds_test = load_from_disk(os.path.join(args.data_dir, "ds_test"))

    existing = load_existing(args.out_csv) if args.skip_existing else set()
    baselines = args.models or DEFAULT_BASELINES
    shared = shared_hps(args)

    if args.litdd_model_path:
        label = "LitDD-BERT (fine-tuned)"
        if label not in existing:
            row = evaluate_only(args.litdd_model_path, label, ds_test)
            append_row(args.out_csv, row)
            print("->", row)

    for name in baselines:
        if name in existing:
            print(f"[SKIP] {name} already in {args.out_csv}")
            continue
        try:
            if args.cv_hp_search:
                hp = hp_search_for_model(name, args)
            elif shared is not None:
                hp = shared
            else:
                hp = {}  # use script defaults
            row = fine_tune_and_eval(name, hp, args, ds_train, ds_test)
            append_row(args.out_csv, row)
            print("->", row)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            append_row(args.out_csv, {"model": name, "precision": "", "recall": "", "f1": ""})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
