#!/usr/bin/env python3
"""Fine-tune the LitDD-BERT classifier and evaluate on the held-out test set.

Methodology (with ``cross_validation/cv_hp_search_bert.py``):

  1. Build the 80/20 train/test split with ``final_traintest_dataset.py``.
  2. **Hyperparameter selection** via ``cv_hp_search_bert.py``: 5-fold
     ``StratifiedGroupKFold`` on the *training* set only. Each (lr,
     weight_decay, ...) combination is scored by mean fold F1.
  3. **Refit** on the *full* training set with the selected hyperparameters
     (this script).
  4. **Evaluate once** on the untouched test set (this script).

Pass the chosen hyperparameters via CLI flags, or use ``--hp_json
hp_search_results.json`` to load them from the search-script output.
"""
import argparse
import json
import os

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

# ModernBERT context is 8,192. The previous 512 cap was inherited from the BERT-large
# base and truncated ~1% of abstracts; keep train and inference lengths equal.
MAX_LENGTH = 8192
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def maybe_load_hp_json(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    # Expect either {"best": {...}} or directly a flat dict.
    return data.get("best", data)


def main(args):
    hp = maybe_load_hp_json(args.hp_json)
    learning_rate = hp.get("learning_rate", args.learning_rate)
    weight_decay = hp.get("weight_decay", args.weight_decay)
    epochs = int(hp.get("epochs", args.epochs))
    train_bs = int(hp.get("train_bs", args.train_bs))
    if hp:
        print(f"[Info] using HPs from {args.hp_json}: lr={learning_rate} wd={weight_decay} "
              f"epochs={epochs} train_bs={train_bs}")

    ds_train = load_from_disk(args.train_ds_dir)
    ds_test = load_from_disk(args.test_ds_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.input_model)

    def preprocess(examples):
        return tokenizer(examples["tiab"], truncation=True, max_length=MAX_LENGTH)

    keep = {"tiab", "label"}

    def tokenize(ds):
        return ds.map(
            preprocess,
            batched=True,
            remove_columns=[c for c in ds.column_names if c not in keep],
        )

    tokenized_train = tokenize(ds_train)
    tokenized_test = tokenize(ds_test)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "eval_accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "eval_precision": prec.compute(predictions=preds, references=labels)["precision"],
            "eval_recall": rec.compute(predictions=preds, references=labels)["recall"],
            "eval_f1": f1.compute(predictions=preds, references=labels)["f1"],
        }

    model = AutoModelForSequenceClassification.from_pretrained(args.input_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="no",          # no peeking at the test set during training
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        report_to=[],
        logging_steps=100,
        dataloader_num_workers=max(1, os.cpu_count() // 2),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    print(f"[Info] Train={len(ds_train)} Test={len(ds_test)} (test held out until final eval)")
    trainer.train()
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)

    test_metrics = trainer.evaluate(tokenized_test)
    print("[TEST metrics]", test_metrics)
    trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_ds_dir", default="train_test/ds_bert_train")
    p.add_argument("--test_ds_dir", default="train_test/ds_test")
    p.add_argument("--input_model", default="answerdotai/ModernBERT-large")
    # Hyperparameter overrides — typically loaded via --hp_json from CV search.
    p.add_argument("--learning_rate", type=float, default=1.736e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--hp_json", default=None,
                   help="JSON file with selected HPs (output of cv_hp_search_bert.py).")
    p.add_argument("--output_dir", default="train_test/bert_finetune_results")
    p.add_argument("--best_model_dir", default="train_test/lit_dd_BERT_best")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
