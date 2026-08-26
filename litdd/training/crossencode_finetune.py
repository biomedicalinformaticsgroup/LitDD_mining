#!/usr/bin/env python3
"""Fine-tune the cross-encoder reranker and evaluate once on the held-out test set.

The cross-encoder takes a ``(tiab, thread)`` text pair, joins them with ``[SEP]``,
and runs a single transformer with a 1-logit classification head. ``model.predict``
returns a sigmoid-bounded relevance score in ``[0, 1]``, which is what downstream
``litdd/pipeline/crossencode.py`` retains per TIAB. Pair order matters — a
cross-encoder is not symmetric — so training, this evaluation, and deployment all
use ``(tiab, thread)``.

Methodology (with ``litdd/training/cv_hp_search_crossencoder.py``):

  1. Build the pair datasets (``build_crossencoder_dataset.py``) and mine hard
     negatives on the train portion (``mine_hard_negatives.py``) — positives only
     are offered to the miner; annotated negatives join as labeled negatives.
  2. **Hyperparameter selection** via ``cv_hp_search_crossencoder.py``:
     5-fold ``StratifiedGroupKFold`` on the *training* set only.
  3. **Refit** on the full training set at the selected hyperparameters, once per
     seed in ``--seeds`` — ``set_seed`` runs BEFORE model construction so every
     source of training randomness is governed by it (this script).
  4. **Evaluate once** on the untouched test set: fixed-0.5-threshold metrics,
     confusion matrix, and a per-item prediction dump per seed for
     ``litdd/evaluation/compare_models.py`` (this script). No threshold is tuned
     on the test set; the deployment gate is calibrated separately
     (``threshold sweep``) and reported, not selected here.

Precision: ``--precision fp32|fp16|bf16`` sets the *training* arithmetic. The
screen showed bf16-vs-fp32 training moves test F1 by ~0.012, so the two are not
interchangeable — train both, compare with McNemar, then decide. ``--score_dtypes``
additionally rescores the test set with the trained model cast to each listed
dtype, measuring inference-precision score drift (deployment's ``dtype=auto``
picks bf16 on H100).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def score_pairs(model, tiabs, threads, batch_size: int) -> np.ndarray:
    pairs = list(zip(tiabs, threads))
    return np.asarray(model.predict(pairs, batch_size=batch_size)).reshape(-1)


def metrics_at_half(labels: np.ndarray, scores: np.ndarray) -> dict:
    preds = (scores >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    return {"f1": round(float(f1), 6), "precision": round(float(prec), 6),
            "recall": round(float(rec), 6),
            "ap": round(float(average_precision_score(labels, scores)), 6),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def train_one_seed(args, hp: dict, seed: int, ds_train, ds_test) -> tuple[dict, "object"]:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import (
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from transformers import set_seed

    set_seed(seed)  # BEFORE model construction — seeding after from_pretrained leaves
    # any freshly initialised parameters outside the seed's control (this bit the screen).
    model = CrossEncoder(args.input_model)
    loss = BinaryCrossEntropyLoss(model)

    train_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=int(hp["epochs"]),
        per_device_train_batch_size=int(hp["train_bs"]),
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=float(hp["learning_rate"]),
        warmup_ratio=float(hp["warmup_ratio"]),
        eval_strategy="no",     # no peeking at the test set during training
        save_strategy="no",
        seed=seed,
        data_seed=seed,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        report_to=[],
    )
    trainer = CrossEncoderTrainer(model=model, args=train_args,
                                  train_dataset=ds_train, loss=loss)
    trainer.train()

    labels = np.asarray(ds_test["label"]).astype(int)
    scores = score_pairs(model, ds_test["tiab"], ds_test["g2p_lgmde"], args.eval_bs)
    m = metrics_at_half(labels, scores)
    m.update({"seed": seed, "precision_mode": args.precision})

    if args.pred_dir:
        os.makedirs(args.pred_dir, exist_ok=True)
        tag = f"{args.run_name}_{args.precision}_seed{seed}"
        with open(os.path.join(args.pred_dir, f"{tag}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "pred", "score"])
            for l, s in zip(labels, scores):
                w.writerow([int(l), int(s >= 0.5), f"{float(s):.6f}"])

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return m, model


def rescore_dtypes(args, seed: int, ds_test, results: list[dict]) -> None:
    """Inference-precision drift: reload the saved model in each dtype and rescore."""
    from sentence_transformers import CrossEncoder

    model_dir = os.path.join(args.save_dir, f"seed_{seed}")
    labels = np.asarray(ds_test["label"]).astype(int)
    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    for name in args.score_dtypes:
        model = CrossEncoder(model_dir, model_kwargs={"torch_dtype": dt[name]})
        scores = score_pairs(model, ds_test["tiab"], ds_test["g2p_lgmde"], args.eval_bs)
        m = metrics_at_half(labels, scores)
        m.update({"seed": seed, "precision_mode": f"{args.precision}-scored-{name}"})
        results.append(m)
        print(f"  [score-dtype {name}] {m}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def maybe_load_hp_json(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get("best", data)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", default="data", help="Directory containing the ds_* subdirs.")
    p.add_argument("--hard_negatives_subdir", default="hard_negatives_dataset")
    p.add_argument("--test_subdir", default="ds_test")
    p.add_argument("--input_model", default="ncbi/MedCPT-Cross-Encoder")
    p.add_argument("--output_dir", default="litdd/training/finetuned_cross_encoders")
    p.add_argument("--save_dir", default=None,
                   help="Save each seed's model under <save_dir>/seed_<n>.")
    p.add_argument("--run_name", default="crossencoder")
    p.add_argument("--seeds", default="42",
                   help="Comma-separated seeds; metrics are reported per seed and mean±sd.")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--score_dtypes", nargs="*", default=[],
                   help="Additionally rescore the test set with the saved model cast to "
                        "these dtypes (fp32/fp16/bf16); requires --save_dir.")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--train_bs", type=int, default=16)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--hp_json", default=None,
                   help="JSON file with selected HPs (output of cv_hp_search_crossencoder.py).")
    p.add_argument("--out_csv", default=None, help="Per-seed metrics + confusion matrix.")
    p.add_argument("--pred_dir", default=None,
                   help="Per-item (label, pred, score) dumps for compare_models.py.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    hp = maybe_load_hp_json(args.hp_json)
    hp = {"learning_rate": hp.get("learning_rate", args.learning_rate),
          "epochs": hp.get("epochs", args.epochs),
          "train_bs": hp.get("train_bs", args.train_bs),
          "warmup_ratio": hp.get("warmup_ratio", args.warmup_ratio)}
    print(f"[Info] HPs: {hp} | precision={args.precision}")
    if args.score_dtypes and not args.save_dir:
        raise SystemExit("--score_dtypes requires --save_dir")

    ds_train = load_from_disk(os.path.join(args.data_dir, args.hard_negatives_subdir))
    ds_test = load_from_disk(os.path.join(args.data_dir, args.test_subdir))
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[Info] train pairs {len(ds_train)} | test pairs {len(ds_test)} | seeds {seeds}")

    results: list[dict] = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        m, model = train_one_seed(args, hp, seed, ds_train, ds_test)
        print(m)
        results.append(m)
        if args.save_dir:
            d = os.path.join(args.save_dir, f"seed_{seed}")
            model.save_pretrained(d)
            print(f"[Info] saved seed {seed} -> {d}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.score_dtypes:
            rescore_dtypes(args, seed, ds_test, results)

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        new = not os.path.exists(args.out_csv)
        fieldnames = ["run_name", "precision_mode", "seed", "f1", "precision", "recall",
                      "ap", "tp", "fp", "fn", "tn"]
        with open(args.out_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new:
                w.writeheader()
            for r in results:
                w.writerow({"run_name": args.run_name, **r})
        print(f"[Info] appended {len(results)} rows -> {args.out_csv}")

    trained = [r for r in results if r["precision_mode"] == args.precision]
    f1s = np.array([r["f1"] for r in trained])
    print(f"\n[Info] test F1 over {len(f1s)} seed(s): "
          f"{f1s.mean():.4f} ± {f1s.std(ddof=1) if len(f1s) > 1 else 0.0:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
