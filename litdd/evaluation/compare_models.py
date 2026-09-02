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


def load(path: str) -> tuple[list[int], list[int]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["label"]) for r in rows], [int(r["pred"]) for r in rows]


def f1(labels, preds) -> float:
    tp = sum(1 for lab, p in zip(labels, preds) if lab == 1 and p == 1)
    fp = sum(1 for lab, p in zip(labels, preds) if lab == 0 and p == 1)
    fn = sum(1 for lab, p in zip(labels, preds) if lab == 1 and p == 0)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def mcnemar_exact(labels, a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar. Returns (b_only, c_only, p)."""
    from math import comb

    b_only = sum(1 for lab, x, y in zip(labels, a, b) if (x == lab) and (y != lab))
    c_only = sum(1 for lab, x, y in zip(labels, a, b) if (x != lab) and (y == lab))
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


def cochran_q(labels: list[int], preds_by_model: dict[str, list[int]]) -> tuple[float, int, float]:
    """Omnibus test across k >= 2 classifiers on the same items (Cochran 1950).

    McNemar only compares two models. Testing every pair instead inflates the false-positive
    rate, and picking the top two by observed score and testing only those is worse -- the
    pair was chosen after seeing the results. Cochran's Q asks one question first: do these k
    models differ at all? Only if it rejects is it legitimate to look at pairs, and then with
    a multiple-comparison correction.

    Q is chi-square distributed with k-1 df under the null that all k perform equally.
    """
    from scipy.stats import chi2

    names = list(preds_by_model)
    k = len(names)
    correct = [[1 if p == lab else 0 for lab, p in zip(labels, preds_by_model[n])] for n in names]
    col = [sum(c) for c in correct]                       # per-model correct counts
    row = [sum(c[i] for c in correct) for i in range(len(labels))]  # models correct per item
    num = (k - 1) * (k * sum(g * g for g in col) - sum(col) ** 2)
    den = k * sum(row) - sum(r * r for r in row)
    if den == 0:
        return 0.0, k - 1, 1.0
    q = num / den
    return q, k - 1, float(chi2.sf(q, k - 1))


def holm(pvals: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Holm-Bonferroni adjusted p-values: controls family-wise error without Bonferroni's
    conservatism, and needs no independence assumption (the pairs share a test set)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (pair, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[pair] = running
    return out


def majority(pred_lists: list[list[int]]) -> list[int]:
    return [1 if sum(col) * 2 > len(col) else 0 for col in zip(*pred_lists)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--model_a", default=None, help="substring matching model A's files")
    ap.add_argument("--model_b", default=None)
    ap.add_argument("--models", nargs="+", default=None,
                    help="Three or more model substrings: runs Cochran's Q across all of them, "
                         "then Holm-corrected pairwise McNemar if Q rejects.")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    def collect(tag):
        out = {}
        for p in sorted(glob.glob(os.path.join(args.pred_dir, "*.csv"))):
            if tag in os.path.basename(p):
                seed = os.path.basename(p).split("seed")[-1].split(".")[0]
                out[seed] = load(p)
        return out

    if args.models:
        tags = args.models
        got = {t: collect(t) for t in tags}
        missing = [t for t, v in got.items() if not v]
        if missing:
            raise SystemExit(f"no predictions found for: {missing}")
        shared = sorted(set.intersection(*(set(v) for v in got.values())))
        if not shared:
            raise SystemExit("models share no common seed")
        labels = got[tags[0]][shared[0]][0]
        maj = {t: majority([got[t][s][1] for s in shared]) for t in tags}

        print(f"{len(tags)} models, seeds {', '.join(shared)}, majority vote over seeds\n")
        for t in sorted(tags, key=lambda t: f1(labels, maj[t]), reverse=True):
            print(f"  {f1(labels, maj[t]):.4f}  {t}")

        q, df, p = cochran_q(labels, maj)
        print(f"\nCochran's Q = {q:.3f}, df = {df}, p = {p:.4f}")
        if p > 0.05:
            print("\n-> These models are NOT distinguishable on this test set. Pairwise tests\n"
                  "   are not licensed; choose on other grounds (context length, size, speed).")
            return 0

        print("\n-> Q rejects: at least one model differs. Holm-corrected pairwise McNemar:\n")
        raw = {}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                raw[(a, b)] = mcnemar_exact(labels, maj[a], maj[b])[2]
        adj = holm(raw)
        print(f"  {'pair':<70} {'raw p':>8} {'Holm p':>8}")
        for (a, b), pa in sorted(adj.items(), key=lambda kv: kv[1]):
            mark = " *" if pa < 0.05 else ""
            print(f"  {a[:33]:<34} vs {b[:33]:<34} {raw[(a,b)]:8.4f} {pa:8.4f}{mark}")
        return 0

    if not (args.model_a and args.model_b):
        raise SystemExit("pass --models for 3+, or --model_a and --model_b for a pair")

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
