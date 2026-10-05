#!/usr/bin/env python3
"""Test whether screens differ significantly on the test set.

Reads per-item prediction CSVs (``label``, ``pred``) written by ``run_bert_benchmark.py
--pred_dir``, one per model and seed. For two models: exact McNemar on the discordant items
and a percentile bootstrap interval on the F1 difference, per shared seed and on the
majority vote over seeds. For three or more models: Cochran's Q across all of them, then
Holm-corrected pairwise McNemar when Q rejects.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

from litdd.evaluation.common import bootstrap_f1_diff, f1_binary, mcnemar_exact


def load(path: str) -> tuple[list[int], list[int]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [int(r["label"]) for r in rows], [int(r["pred"]) for r in rows]


def cochran_q(labels: list[int], preds_by_model: dict[str, list[int]]) -> tuple[float, int, float]:
    """Cochran's Q across k classifiers on the same items: statistic, degrees of freedom, p."""
    from scipy.stats import chi2

    names = list(preds_by_model)
    k = len(names)
    correct = [[1 if p == lab else 0 for lab, p in zip(labels, preds_by_model[n])] for n in names]
    col = [sum(c) for c in correct]
    row = [sum(c[i] for c in correct) for i in range(len(labels))]
    num = (k - 1) * (k * sum(g * g for g in col) - sum(col) ** 2)
    den = k * sum(row) - sum(r * r for r in row)
    if den == 0:
        return 0.0, k - 1, 1.0
    q = num / den
    return q, k - 1, float(chi2.sf(q, k - 1))


def holm(pvals: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Holm-Bonferroni adjusted p-values."""
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
                    help="three or more model substrings: Cochran's Q across all of them, "
                         "then Holm-corrected pairwise McNemar if Q rejects")
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
        for t in sorted(tags, key=lambda t: f1_binary(labels, maj[t]), reverse=True):
            print(f"  {f1_binary(labels, maj[t]):.4f}  {t}")

        q, df, p = cochran_q(labels, maj)
        print(f"\nCochran's Q = {q:.3f}, df = {df}, p = {p:.4f}")
        if p > 0.05:
            print("\nThe models are not distinguishable on this test set; pairwise tests are not reported.")
            return 0

        print("\nQ rejects. Holm-corrected pairwise McNemar:\n")
        raw = {}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                raw[(a, b)] = mcnemar_exact(labels, maj[a], maj[b])[2]
        adj = holm(raw)
        print(f"  {'pair':<70} {'raw p':>8} {'Holm p':>8}")
        for (a, b), pa in sorted(adj.items(), key=lambda kv: kv[1]):
            mark = " *" if pa < 0.05 else ""
            print(f"  {a[:33]:<34} vs {b[:33]:<34} {raw[(a, b)]:8.4f} {pa:8.4f}{mark}")
        return 0

    if not (args.model_a and args.model_b):
        raise SystemExit("pass --models for three or more models, or --model_a and --model_b for a pair")

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
        print(f"{s:>6}  {f1_binary(labels, a):7.4f} {f1_binary(labels, b):7.4f} {d:+8.4f}  {bo:>5} {co:>5} "
              f"{p:10.4f}  [{lo:+.4f}, {hi:+.4f}]")

    labels = A[shared[0]][0]
    ma, mb = majority([A[s][1] for s in shared]), majority([B[s][1] for s in shared])
    bo, co, p = mcnemar_exact(labels, ma, mb)
    d, lo, hi = bootstrap_f1_diff(labels, ma, mb, args.n_boot)
    print(f"\n{'major':>6}  {f1_binary(labels, ma):7.4f} {f1_binary(labels, mb):7.4f} {d:+8.4f}  {bo:>5} {co:>5} "
          f"{p:10.4f}  [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
