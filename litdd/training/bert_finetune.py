#!/usr/bin/env python3
"""Fine-tune a screen classifier on a training dataset and evaluate it once on the test set.

Reads two HuggingFace ``save_to_disk`` directories (``--train_ds_dir``, ``--test_ds_dir``) with
``tiab`` and ``label`` columns, and the hyperparameters either from the flags or from the JSON
written by ``cv_hp_search_bert.py`` (``--hp_json``, whose ``best`` entry overrides the flags).
Trains ``--input_model`` on the full training set, saves the model and tokenizer to
``--best_model_dir``, evaluates on the test set and writes the test metrics (accuracy,
precision, recall, F1) next to the trainer output.
"""
from __future__ import annotations

import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from litdd.training.screen_common import make_compute_metrics, maybe_load_hp_json

# ModernBERT context length; training and inference use the same cap.
MAX_LENGTH = 8192
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main(args: argparse.Namespace) -> None:
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
    model = AutoModelForSequenceClassification.from_pretrained(args.input_model, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="no",  # the test set is evaluated once, after training
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
        compute_metrics=make_compute_metrics(accuracy=True),
    )

    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))

    print(f"[Info] Train={len(ds_train)} Test={len(ds_test)}")
    trainer.train()
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)

    test_metrics = trainer.evaluate(tokenized_test)
    print("[TEST metrics]", test_metrics)
    trainer.save_metrics("test", test_metrics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train_ds_dir", default="litdd/training/ds_bert_train")
    p.add_argument("--test_ds_dir", default="litdd/training/ds_test")
    p.add_argument("--input_model", default="answerdotai/ModernBERT-large")
    # Hyperparameters; --hp_json overrides these with the CV-selected values.
    p.add_argument("--learning_rate", type=float, default=1.736e-5)
    p.add_argument("--train_bs", type=int, default=32)
    p.add_argument("--eval_bs", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--hp_json", default=None,
                   help="JSON file with selected HPs (output of cv_hp_search_bert.py).")
    p.add_argument("--output_dir", default="litdd/training/bert_finetune_results")
    p.add_argument("--best_model_dir", default="litdd/training/lit_dd_BERT_best")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
