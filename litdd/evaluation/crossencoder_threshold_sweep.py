#!/usr/bin/env python3
"""Precision/recall/F1 vs score threshold for the cross-encoder gate (R3.6).

Reads the per-item ``label,pred,score`` dumps written by
``litdd/training/crossencode_finetune.py --pred_dir`` (one file per seed) and sweeps
the deployment gate over ``--thresholds``. With several seeds, each threshold row
reports mean ± sd across seeds, so the recalibrated gate is chosen from an
expectation rather than one draw.

The output is the evidence for the reported operating point: the gate is *read off*
this curve and stated with its precision/recall trade-off — it is not tuned into the
model, and test-set metrics elsewhere stay at the fixed 0.5 threshold.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np


def load_scores(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    labels = np.array([int(r["label"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    return labels, scores


def prf_at(labels: np.ndarray, scores: np.ndarray, t: float) -> tuple[float, float, float, int]:
    preds = scores >= t
    tp = int((preds & (labels == 1)).sum())
    fp = int((preds & (labels == 0)).sum())
    fn = int((~preds & (labels == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp + fp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pred_glob", required=True,
                   help="Glob over the per-seed prediction dumps, e.g. "
                        "'revision/crossencoder/preds/final_fp16_seed*.csv'")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[round(0.05 * k, 2) for k in range(1, 20)],
                   help="Default: 0.05 … 0.95 in 0.05 steps.")
    p.add_argument("--out_csv", default="revision/crossencoder_threshold_sweep.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(glob.glob(args.pred_glob))
    if not files:
        raise SystemExit(f"no prediction files match {args.pred_glob}")
    runs = [load_scores(f) for f in files]
    print(f"[Info] {len(files)} seed dump(s): {[os.path.basename(f) for f in files]}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "n_seeds", "precision_mean", "precision_sd",
                    "recall_mean", "recall_sd", "f1_mean", "f1_sd", "n_passing_mean"])
        for t in args.thresholds:
            stats = np.array([prf_at(lab, sc, t) for lab, sc in runs])
            m = stats.mean(axis=0)
            s = stats.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(4)
            w.writerow([t, len(runs), round(m[0], 4), round(s[0], 4), round(m[1], 4),
                        round(s[1], 4), round(m[2], 4), round(s[2], 4), round(m[3], 1)])
            print(f"  t={t:.2f}  P={m[0]:.3f}±{s[0]:.3f}  R={m[1]:.3f}±{s[1]:.3f}  "
                  f"F1={m[2]:.3f}±{s[2]:.3f}")
    print(f"[Info] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
