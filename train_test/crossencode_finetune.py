#!/usr/bin/env python3
"""Fine-tune the cross-encoder reranker and evaluate on the held-out test set.

The cross-encoder takes a `(query, candidate)` text pair, joins them with
``[SEP]``, and runs a single transformer with a 1-logit classification head.
``model.predict`` returns a sigmoid-bounded relevance score in `[0, 1]`,
which is what downstream ``annotate_pubmed/crossencode.py`` retains as the
top-5 score per TIAB.

Methodology (with ``cross_validation/cv_hp_search_crossencoder.py``):

  1. Build the 80/20 train/test split (``final_traintest_dataset.py``)
     and mine hard negatives on the train portion (``mine_hard_negatives.py``).
  2. **Hyperparameter selection** via ``cv_hp_search_crossencoder.py``:
     5-fold ``StratifiedGroupKFold`` on the *training* set only, scored by
     mean fold AUC / binary-F1.
  3. **Refit** on the *full* hard-negatives training set with the selected
     hyperparameters (this script).
  4. **Evaluate once** on the untouched test set (this script).

Pass the chosen hyperparameters via CLI flags, or use ``--hp_json
hp_search_results.json`` to load them from the search-script output.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True


def maybe_load_hp_json(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("best", data)


def make_evaluator(ds, name: str) -> CrossEncoderClassificationEvaluator:
    return CrossEncoderClassificationEvaluator(
        sentence_pairs=list(zip(ds["g2p_lgmde"], ds["tiab"])),
        labels=list(ds["label"]),
        name=name,
    )


def finetune_crossencoder(args) -> None:
    hp = maybe_load_hp_json(args.hp_json)
    learning_rate = hp.get("learning_rate", args.learning_rate)
    epochs = int(hp.get("epochs", args.epochs))
    train_bs = int(hp.get("train_bs", args.train_bs))
    warmup_ratio = hp.get("warmup_ratio", args.warmup_ratio)
    if hp:
        print(f"[Info] using HPs from {args.hp_json}: lr={learning_rate} epochs={epochs} "
              f"train_bs={train_bs} warmup_ratio={warmup_ratio}")

    hard_negatives_ds = load_from_disk(os.path.join(args.data_dir, args.hard_negatives_subdir))
    ds_test = load_from_disk(os.path.join(args.data_dir, args.test_subdir))

    model = CrossEncoder(args.input_model)
    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        eval_strategy="no",     # no peeking at the test set during training
        save_strategy="no",
        report_to=[],
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=train_args,
        train_dataset=hard_negatives_ds,
        loss=loss,
    )
    trainer.train()
    trainer.save_model(args.best_model_dir)

    print(f"\n[Info] Held-out test set ({args.input_model}):")
    test_eval = make_evaluator(ds_test, name="anno_test")
    print(test_eval(model))
    print(f"[Info] saved fine-tuned cross-encoder → {args.best_model_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", default="train_test", help="Directory containing the ds_* subdirs.")
    p.add_argument("--hard_negatives_subdir", default="hard_negatives_dataset")
    p.add_argument("--test_subdir", default="ds_test")
    p.add_argument("--input_model", default="ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--output_dir", default="train_test/finetuned_cross_encoders")
    p.add_argument("--best_model_dir", default="train_test/finetuned_ncbi_medcpt_cross")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3.879271032713091e-05)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--hp_json", default=None,
                   help="JSON file with selected HPs (output of cv_hp_search_crossencoder.py).")
    return p.parse_args()


if __name__ == "__main__":
    finetune_crossencoder(parse_args())
