#!/usr/bin/env python3
"""Test whether two screens differ significantly on the test set (R1.3 base selection).

Aggregate F1 cannot answer "is BiomedBERT actually better than BioClinical?" -- 0.9265 vs
0.9217 on 2,779 shared items may be a handful of flipped predictions, and the seed spread is
of the same order as the gap. Two paired tests, both on the same items:

* **McNemar** on the discordant pairs (b = A right/B wrong, c = A wrong/B right). Exact
  binomial, so it is valid at the small counts these comparisons produce. This is the standard
  test for two classifiers evaluated on one test set.
* **Bootstrap CI** on the F1 difference, resampling items. If the interval spans 0 the models
  are not separated by this test set, whatever the point estimates say.

Predictions come from `run_bert_benchmark.py --pred_dir`. With several seeds per model the
comparison is run per seed pair and also on the majority vote, since a single seed cannot
distinguish a real difference from an initialisation draw.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import random
from collections import defaultdict


def load(path: str) -> tuple[list[int], list[int]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["label"]) for r in rows], [int(r["pred"]) for r in rows]


def f1(labels, preds) -> float:
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def mcnemar_exact(labels, a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar. Returns (b_only, c_only, p)."""
    from math import comb

    b_only = sum(1 for l, x, y in zip(labels, a, b) if (x == l) and (y != l))
    c_only = sum(1 for l, x, y in zip(labels, a, b) if (x != l) and (y == l))
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b_only, c_only, p


def bootstrap_f1_diff(labels, a, b, n_boot=2000, seed=0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        diffs.append(f1(la, [a[i] for i in idx]) - f1(la, [b[i] for i in idx]))
    diffs.sort()
    return (f1(labels, a) - f1(labels, b),
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1])


def majority(pred_lists: list[list[int]]) -> list[int]:
    return [1 if sum(col) * 2 > len(col) else 0 for col in zip(*pred_lists)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--model_a", required=True, help="substring matching model A's files")
    ap.add_argument("--model_b", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    def collect(tag):
        out = {}
        for p in sorted(glob.glob(os.path.join(args.pred_dir, "*.csv"))):
            if tag in os.path.basename(p):
                seed = os.path.basename(p).split("seed")[-1].split(".")[0]
                out[seed] = load(p)
        return out

    A, B = collect(args.model_a), collect(args.model_b)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit(f"no shared seeds: A={sorted(A)} B={sorted(B)}")

    print(f"A = {args.model_a}\nB = {args.model_b}\nshared seeds: {', '.join(shared)}\n")
    print(f"{'seed':>6}  {'F1(A)':>7} {'F1(B)':>7} {'diff':>8}  {'A>B':>5} {'B>A':>5} "
          f"{'McNemar p':>10}  {'95% CI on diff':>24}")
    for s in shared:
        labels, a = A[s]
        _, b = B[s]
        bo, co, p = mcnemar_exact(labels, a, b)
        d, lo, hi = bootstrap_f1_diff(labels, a, b, args.n_boot)
        print(f"{s:>6}  {f1(labels,a):7.4f} {f1(labels,b):7.4f} {d:+8.4f}  {bo:>5} {co:>5} "
              f"{p:10.4f}  [{lo:+.4f}, {hi:+.4f}]")

    labels = A[shared[0]][0]
    ma, mb = majority([A[s][1] for s in shared]), majority([B[s][1] for s in shared])
    bo, co, p = mcnemar_exact(labels, ma, mb)
    d, lo, hi = bootstrap_f1_diff(labels, ma, mb, args.n_boot)
    print(f"\n{'major':>6}  {f1(labels,ma):7.4f} {f1(labels,mb):7.4f} {d:+8.4f}  {bo:>5} {co:>5} "
          f"{p:10.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("\nA CI spanning 0, or p > 0.05, means this test set does not separate the two models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
