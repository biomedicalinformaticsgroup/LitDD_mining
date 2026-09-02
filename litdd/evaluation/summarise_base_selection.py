#!/usr/bin/env python3
"""Summarise the base-model selection sweep: mean +/- sd over seeds (R1.3).

Emits the table in the form it should be reported: test-set precision/recall/F1 with the
confusion matrix, and clean held-out external recall -- the latter measured only on truth
papers that are in neither the annotated train set nor the test set.

Single runs are not comparable here: the bases sit within ~0.02 F1 of each other, which is
inside the seed spread. Fixed-weight checkpoints (the LitDD models) have no seed dimension
and are reported as single rows.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from collections import defaultdict

NUM = ("precision", "recall", "f1", "external_recall_all")
CNT = ("tp", "fp", "fn", "tn")


def fmt(vals: list[float], places: int = 4) -> str:
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{places}f}"
    return f"{st.mean(vals):.{places}f} ± {st.stdev(vals):.{places}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    by_model: dict[str, list[dict]] = defaultdict(list)
    with open(args.in_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("f1"):
                by_model[r["model"]].append(r)

    rows = []
    for model, rs in by_model.items():
        agg = {"model": model, "n_seeds": len(rs),
               "seeds": ";".join(sorted(str(r.get("seed", "")) for r in rs))}
        for k in NUM:
            agg[k] = fmt([float(r[k]) for r in rs if r.get(k)])
        for k in CNT:  # counts are integers; report the mean over seeds
            vals = [int(r[k]) for r in rs if r.get(k)]
            agg[k] = f"{st.mean(vals):.0f}" if vals else "-"
        agg["_sort"] = st.mean([float(r["f1"]) for r in rs])
        rows.append(agg)
    rows.sort(key=lambda r: r.pop("_sort"), reverse=True)

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    w = max(len(r["model"]) for r in rows)
    print(f"{'model':{w}}  {'n':>2} {'test P':>15} {'test R':>15} {'test F1':>15} "
          f"{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}  {'clean held-out recall':>21}")
    for r in rows:
        print(f"{r['model']:{w}}  {r['n_seeds']:>2} {r['precision']:>15} {r['recall']:>15} "
              f"{r['f1']:>15} {r['tp']:>5}{r['fp']:>5}{r['fn']:>5}{r['tn']:>5}  "
              f"{r['external_recall_all']:>21}")
    print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
