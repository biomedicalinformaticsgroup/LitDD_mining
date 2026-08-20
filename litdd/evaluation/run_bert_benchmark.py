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

Inputs:  a training set and ``ds_test``. Prefer ``--train_ds_dir`` pointing at the
         **annotated train set** (``ds_hirecall_train``); ``ds_bert_train`` from
         ``final_traintest_dataset.py`` is deprecated and reproduces pre-revision
         results only.
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
    p.add_argument("--train_ds_dir", default=None,
                   help="Explicit training dataset. The annotated train set "
                        "(ds_hirecall_train) and the held-out test set do not live in one "
                        "directory, so --data_dir's convention cannot address them.")
    p.add_argument("--test_ds_dir", default=None, help="Explicit held-out test dataset.")
    p.add_argument("--data_dir", default="data",
                   help="Directory containing ds_bert_train and ds_test. DEPRECATED for training: "
                        "prefer --train_ds_dir with the annotated train set "
                        "(ds_hirecall_train).")
    p.add_argument("--out_csv", default="results/bert_results.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="Override baseline list.")
    p.add_argument("--litdd_label", default=None,
                   help="Row label for --litdd_model_path (default: the path itself).")
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
    p.add_argument("--skip_baselines", action="store_true",
                   help="Evaluate only --litdd_model_path and skip baseline fine-tuning. "
                        "Turns a multi-GPU-hour job into an inference-only one.")
    p.add_argument("--external_csv", default=None,
                   help="Truth corpus (pmid, tiab, source[, gene]) to score for external "
                        "recall. Every baseline gets the same corpus, so Table 1's F1 "
                        "comparison gains the generalisation comparison it currently lacks.")
    p.add_argument("--external_scope", choices=["raw", "heldout_gene_fold"], default="raw",
                   help="'raw' scores every row -- the unconditioned figure. "
                        "'heldout_gene_fold' restricts to the 10%% gene fold held out of "
                        "training, which is the basis of the previously reported 98.5%% and "
                        "is NOT comparable to 'raw'.")
    p.add_argument("--external_threshold", type=float, default=0.5)
    p.add_argument("--pred_dir", default=None,
                   help="Dump per-example test-set predictions here, one CSV per (model, seed). "
                        "Needed to test whether two models differ significantly: McNemar's test "
                        "works on paired per-item outcomes, which aggregate metrics discard.")
    return p.parse_args()


def load_existing(out_csv: str) -> set[tuple[str, str]]:
    """(model, seed) pairs already recorded, so a resumed multi-seed sweep is not truncated."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {(row["model"], str(row.get("seed", ""))) for row in csv.DictReader(f)
                if row.get("model")}


def append_row(out_csv: str, row: dict) -> None:
    is_new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def tokenize(ds, tokenizer, keep={"tiab", "label"}):
    def fn(b):
        return tokenizer(b["tiab"], truncation=True, max_length=512)
    return ds.map(fn, batched=True,
                  remove_columns=[c for c in ds.column_names if c not in keep])



def dump_predictions(trainer, tok_test, ds_test, pred_dir: str, model_name: str, seed: int):
    """Write per-example test predictions so model pairs can be compared statistically.

    Aggregate F1 cannot say whether two models differ: 0.9265 vs 0.9217 on the same 2,779
    items may be a handful of flipped predictions. McNemar's test needs the paired per-item
    outcomes, which only exist if they are written out at evaluation time.
    """
    import numpy as np

    os.makedirs(pred_dir, exist_ok=True)
    logits = trainer.predict(tok_test).predictions
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(ds_test["label"])
    safe = model_name.replace("/", "__")
    path = os.path.join(pred_dir, f"{safe}__seed{seed}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "pred"])
        for i, (l, p) in enumerate(zip(labels, preds)):
            w.writerow([i, int(l), int(p)])
    print(f"[INFO] wrote {len(preds)} predictions -> {path}", flush=True)


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
            # Raw counts as well as rates: a reviewer asked for the confusion matrix, and
            # precision/recall alone hide how many records each rate is computed over.
            "eval_tp": int(((preds == 1) & (labels == 1)).sum()),
            "eval_fp": int(((preds == 1) & (labels == 0)).sum()),
            "eval_fn": int(((preds == 0) & (labels == 1)).sum()),
            "eval_tn": int(((preds == 0) & (labels == 0)).sum()),
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
    # Seed BEFORE from_pretrained: the classification head is initialised at load time, so
    # seeding afterwards leaves head init uncontrolled (finetune_seeds.py already did this).
    from transformers import set_seed

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
        # fp32 throughout. finetune_seeds.py trained the locked checkpoint in bf16 while this
        # script ran fp32, so "the same protocol" produced numerically different models and the
        # locked model could not be compared to its own base. Standardised on fp32: slower, but
        # irrelevant at this scale and it removes an uncontrolled variable.
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

    row = {
        "model": model_name, "seed": args.seed,
        "precision": round(float(test_metrics["eval_precision"]), 6),
        "recall": round(float(test_metrics["eval_recall"]), 6),
        "f1": round(float(test_metrics["eval_f1"]), 6),
            "tp": int(test_metrics["eval_tp"]), "fp": int(test_metrics["eval_fp"]),
        "fn": int(test_metrics["eval_fn"]), "tn": int(test_metrics["eval_tn"]),
    }
    if getattr(args, "pred_dir", None):
        dump_predictions(trainer, tok_test, ds_test, args.pred_dir, model_name, args.seed)
    if getattr(args, "external_csv", None):
        row.update(external_recall(
            model, tokenizer, args.external_csv, args.external_scope,
            args.external_threshold,
            pred_path=(os.path.join(args.pred_dir,
                                    f"EXT__{model_name.replace('/', '__')}__seed{args.seed}.csv")
                       if args.pred_dir else None)))
    return row


def _b10(gene: str) -> int:
    """Stable 10-way gene fold (matches finetune_seeds.py's held-out definition)."""
    import hashlib
    return int(hashlib.md5(str(gene).encode()).hexdigest(), 16) % 10


def external_recall(model, tokenizer, external_csv: str, scope: str,
                    threshold: float, max_length: int | None = None,
                    pred_path: str | None = None) -> dict:
    """Recall on an external truth corpus, per source and overall.

    Reported alongside test F1 because a screen can look strong on a held-out split of its
    own annotation distribution and still generalise poorly to independently curated
    literature -- which is the concern R3.4 is about. Baselines were previously compared on
    test F1 only.
    """
    import numpy as np
    import pandas as pd
    import torch

    # Cap at whatever the model actually supports. ModernBERT allows 8,192 but the classic
    # BERT baselines (BioBERT, BiomedBERT) have 512 learned position embeddings, and feeding
    # them longer sequences fails with "size of tensor a (515) must match tensor b (512)".
    limit = getattr(model.config, "max_position_embeddings", 512) or 512
    limit = min(limit, getattr(tokenizer, "model_max_length", limit) or limit)
    max_length = limit if max_length is None else min(max_length, limit)

    ext = pd.read_csv(external_csv, dtype=str).drop_duplicates("pmid")
    if scope == "heldout_gene_fold":
        if "gene" not in ext.columns:
            raise SystemExit("--external_scope heldout_gene_fold needs a 'gene' column")
        gb = ext.groupby("pmid")["gene"].apply(lambda gs: {_b10(g) for g in gs})
        ext = ext[ext["pmid"].map(gb).map(lambda f: f == {0})].reset_index(drop=True)
    texts = ext["tiab"].fillna("").tolist()

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i:i + 64], truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(model.device)
            probs.extend(torch.softmax(model(**enc).logits, -1)[:, 1].float().cpu().tolist())
    probs = np.asarray(probs)

    out = {"external_scope": scope, "external_n": len(ext),
           "external_recall_all": round(float((probs >= threshold).mean()), 4)}
    if pred_path:
        # Per-paper calls, so external recall can be compared against another system on the
        # same papers with a paired test rather than by eyeballing two rates.
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


def evaluate_only(model_name: str, label: str, ds_test, external_csv=None,
                  external_scope="raw", external_threshold=0.5, seed: int = 42,
                  pred_dir: str | None = None) -> dict:
    """Evaluate a checkpoint as-is, with no fine-tuning.

    For an already-fine-tuned model this scores the shipped weights. For a *base* model it
    attaches a freshly-initialised classification head and scores that -- i.e. what the
    pretrained encoder gives you before any task training. That head is random, so the seed
    is fixed here: without it the numbers are an arbitrary draw and not reproducible.
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
        compute_metrics=make_compute_metrics(),
    )
    metrics = trainer.evaluate(tok_test)
    row = {
        "model": label, "seed": seed,
        "precision": round(float(metrics["eval_precision"]), 6),
        "recall": round(float(metrics["eval_recall"]), 6),
        "f1": round(float(metrics["eval_f1"]), 6),
            "tp": int(metrics["eval_tp"]), "fp": int(metrics["eval_fp"]),
        "fn": int(metrics["eval_fn"]), "tn": int(metrics["eval_tn"]),
    }
    if external_csv:
        row.update(external_recall(
            model, tokenizer, external_csv, external_scope, external_threshold,
            pred_path=(os.path.join(pred_dir, f"EXT__{label.replace('/', '__')}.csv")
                       if pred_dir else None)))
    return row


def shared_hps(args) -> dict | None:
    if args.hp_json:
        with open(args.hp_json) as f:
            data = json.load(f)
        return data.get("best", data)
    return None


def main() -> int:
    args = parse_args()
    # ds_bert_train is only needed to fine-tune baselines; --skip_baselines must not
    # require training data it never touches.
    train_path = args.train_ds_dir or os.path.join(args.data_dir, "ds_bert_train")
    test_path = args.test_ds_dir or os.path.join(args.data_dir, "ds_test")
    ds_train = None if args.skip_baselines else load_from_disk(train_path)
    ds_test = load_from_disk(test_path)
    print(f"train: {train_path}\ntest : {test_path}", flush=True)

    existing = load_existing(args.out_csv) if args.skip_existing else set()
    baselines = [] if args.skip_baselines else (args.models or DEFAULT_BASELINES)
    shared = shared_hps(args)

    if args.litdd_model_path:
        # Name the row after the checkpoint, not a fixed string: several checkpoints are
        # commonly scored into one CSV and a constant label makes the rows indistinguishable
        # except by run order.
        label = args.litdd_label or f"eval-only: {args.litdd_model_path}"
        if (label, str(args.seed)) not in existing:
            row = evaluate_only(args.litdd_model_path, label, ds_test,
                                external_csv=args.external_csv,
                                external_scope=args.external_scope,
                                external_threshold=args.external_threshold,
                                seed=args.seed, pred_dir=args.pred_dir)
            append_row(args.out_csv, row)
            print("->", row)

    for name in baselines:
        if (name, str(args.seed)) in existing:
            print(f"[SKIP] {name} seed={args.seed} already in {args.out_csv}")
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
            append_row(args.out_csv, {"model": name, "seed": args.seed,
                                      "precision": "", "recall": "", "f1": ""})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
