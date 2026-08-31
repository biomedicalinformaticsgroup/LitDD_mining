#!/usr/bin/env python3
"""Score a pair dataset with any cross-encoder checkpoint and dump per-item predictions.

Produces the same ``label,pred,score`` file that ``crossencode_finetune.py --pred_dir``
writes for freshly trained seeds, so untrained baselines (``ncbi/MedCPT-Cross-Encoder``) and
previously released checkpoints (``tmy100000001/LitDD_crossencoder``) can enter the same
``compare_models.py`` Cochran's Q / McNemar comparison as the re-finetuned models. Pairs are
scored in deployment order ``(tiab, thread)`` at a fixed 0.5 threshold; ``--dtype`` defaults
to fp32 (the reporting standard).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="HF id or local checkpoint directory.")
    p.add_argument("--ds_dir", required=True, help="save_to_disk dataset with tiab/g2p_lgmde/label.")
    p.add_argument("--out_csv", required=True, help="Per-item label,pred,score dump.")
    p.add_argument("--metrics_csv", default=None, help="Append one summary row here.")
    p.add_argument("--run_name", default=None)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from sentence_transformers import CrossEncoder

    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = CrossEncoder(args.model, model_kwargs={"torch_dtype": dt})
    ds = load_from_disk(args.ds_dir)
    pairs = list(zip(ds["tiab"], ds["g2p_lgmde"]))
    scores = np.asarray(model.predict(pairs, batch_size=args.batch_size)).reshape(-1)
    labels = np.asarray(ds["label"]).astype(int)
    preds = (scores >= 0.5).astype(int)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "pred", "score"])
        for lab, sc in zip(labels, scores):
            w.writerow([int(lab), int(sc >= 0.5), f"{float(sc):.6f}"])

    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary",
                                                       zero_division=0)
    row = {"run_name": args.run_name or args.model, "precision_mode": f"scored-{args.dtype}",
           "seed": "", "f1": round(float(f1), 6), "precision": round(float(prec), 6),
           "recall": round(float(rec), 6),
           "ap": round(float(average_precision_score(labels, scores)), 6)
           if labels.any() and not labels.all() else "",
           "tp": int(((preds == 1) & (labels == 1)).sum()),
           "fp": int(((preds == 1) & (labels == 0)).sum()),
           "fn": int(((preds == 0) & (labels == 1)).sum()),
           "tn": int(((preds == 0) & (labels == 0)).sum())}
    print(row)
    if args.metrics_csv:
        new = not os.path.exists(args.metrics_csv)
        with open(args.metrics_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    print(f"[Info] wrote {args.out_csv} ({len(labels)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
