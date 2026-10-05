#!/usr/bin/env python3
"""Merge the screen base-model sweep CSVs into one table with test metrics and external recall.

Later inputs override earlier ones for the same model; rows without an F1 value are skipped.
The training set of each model is carried as a column.
"""
from __future__ import annotations

import argparse
import csv
import os

TRAIN_SET = {
    "LitDD-BERT (fine-tuned)": "augmented annotation set",
}
DEFAULT_TRAIN = "original annotation set"


def load(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["model"]: r for r in csv.DictReader(f) if r.get("f1")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    merged: dict[str, dict] = {}
    for p in args.inputs:
        merged.update(load(p))

    rows = []
    for model, r in merged.items():
        rows.append({
            "model": model,
            "fine_tuned_on": TRAIN_SET.get(model, DEFAULT_TRAIN),
            "test_precision": r["precision"], "test_recall": r["recall"], "test_f1": r["f1"],
            "raw_truth_n": r.get("external_n", ""),
            "raw_truth_recall": r.get("external_recall_all", ""),
        })
    rows.sort(key=lambda x: float(x["raw_truth_recall"] or 0), reverse=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    w = max(len(r["model"]) for r in rows)
    print(f"{'model':{w}}  {'test F1':>8}  {'raw-truth recall':>17}  fine-tuned on")
    for r in rows:
        print(f"{r['model']:{w}}  {float(r['test_f1']):8.4f}  "
              f"{float(r['raw_truth_recall'] or 0):17.4f}  {r['fine_tuned_on']}")
    print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
