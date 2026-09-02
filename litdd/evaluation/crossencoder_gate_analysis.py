#!/usr/bin/env python3
"""Why 0.9? Threshold analysis for the cross-encoder gate on all three evaluation axes (R3.6).

The gate's job in the cascade is to be a *precision* filter ahead of the LLM adjudication step,
so plain F1 is the wrong selection criterion: it is maximised at 0.5 on both the in-domain
test set and the frozen benchmark, where the gate would pass the most false candidates. This
script lays out, per threshold and averaged over seeds:

  * in-domain test precision / recall / F1 (fixed labels, 2,771 pairs);
  * frozen external benchmark: in-scope pair recall, deployment-faithful paper recall
    (truth entry among the abstract's gate candidates and scored >= t), false-fire rate on
    the 87,600 corpus abstracts, and the benchmark precision lower bound at its ~2%
    prevalence, with F1 and the precision-weighted F0.5;
  * the marginal trade: in-scope recall lost per false-fire abstract avoided between
    consecutive thresholds, and the same cost relative to the 0.5-0.9 baseline slope.

The selection rule stated in the reviewer response: choose the highest threshold before the
marginal recall cost per avoided false fire departs from its low-threshold slope (the knee),
subject to the precision-weighted F0.5 being within one seed-sd of its maximum. Inputs are
the per-pair score dumps written by ``crossencoder_frozen_eval.py`` and the per-item test
dumps from ``crossencode_finetune.py``.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run_prefix", default="ce_flat_addneg_fp32_seed",
                   help="Prefix of the frozen_eval dumps and test preds, one per seed.")
    p.add_argument("--frozen_dir", default="revision/crossencoder/frozen_eval")
    p.add_argument("--pred_dir", default="revision/crossencoder/preds")
    p.add_argument("--frozen_pos_ds", default="revision/crossencoder/flat/ds_frozen_pos")
    p.add_argument("--n_corpus", type=int, default=87600,
                   help="Corpus abstracts in the frozen benchmark (fire-rate denominator).")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99])
    p.add_argument("--out_csv", default="revision/crossencoder_gate_analysis.csv")
    return p.parse_args()


def fbeta(p: float, r: float, beta: float) -> float:
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def main() -> int:
    args = parse_args()
    from datasets import load_from_disk

    meta = load_from_disk(args.frozen_pos_ds).to_pandas()[["pmid", "g2p_id", "truth_in_candidates"]]
    seeds = sorted(int(f.split("seed")[-1].split("__")[0]) for f in
                   glob.glob(os.path.join(args.frozen_dir, f"{args.run_prefix}*__frozen_pos_scores.csv"))
                   if "scored" not in f)
    rows = []
    for seed in seeds:
        pos = pd.read_csv(os.path.join(args.frozen_dir, f"{args.run_prefix}{seed}__frozen_pos_scores.csv"),
                          dtype={"pmid": str, "g2p_id": str}).merge(meta, on=["pmid", "g2p_id"], how="left")
        neg = pd.read_csv(os.path.join(args.frozen_dir, f"{args.run_prefix}{seed}__frozen_neg_max_scores.csv"))
        test = pd.read_csv(os.path.join(args.pred_dir, f"{args.run_prefix}{seed}.csv"))
        dep = pos[pos.truth_in_candidates.fillna(False)]
        insc = pos[pos.scope_category == "in_scope"]
        y, s = test.label.values, test.score.values
        for t in args.thresholds:
            p = s >= t
            tp, fp, fn = int((p & (y == 1)).sum()), int((p & (y == 0)).sum()), int((~p & (y == 1)).sum())
            P, R = tp / (tp + fp), tp / (tp + fn)
            rec_papers = int(dep.assign(h=dep.score >= t).groupby("pmid").h.max().sum())
            n_pos_papers = dep.pmid.nunique()
            fired = int((neg.max_score >= t).sum())
            prec_lb = rec_papers / (rec_papers + fired)
            rec_dep = rec_papers / n_pos_papers
            rows.append({
                "seed": seed, "threshold": t,
                "test_precision": P, "test_recall": R, "test_f1": fbeta(P, R, 1.0),
                "inscope_pair_recall": float((insc.score >= t).mean()),
                "deploy_paper_recall": rec_dep,
                "fired_corpus_abstracts": fired, "fire_pct": 100 * fired / args.n_corpus,
                "bench_precision_lb": prec_lb,
                "bench_f1_lb": fbeta(prec_lb, rec_dep, 1.0),
                "bench_f05_lb": fbeta(prec_lb, rec_dep, 0.5),
            })
    df = pd.DataFrame(rows)
    g = df.groupby("threshold")
    out = g.mean(numeric_only=True).drop(columns=["seed"])
    sd = g.std(numeric_only=True)
    for c in ["test_f1", "inscope_pair_recall", "deploy_paper_recall", "fire_pct",
              "bench_precision_lb", "bench_f1_lb", "bench_f05_lb"]:
        out[f"{c}_sd"] = sd[c]

    # marginal trade between consecutive thresholds: in-scope recall lost per false fire avoided
    out = out.sort_index()
    d_rec = -out["inscope_pair_recall"].diff()
    d_fire = -out["fired_corpus_abstracts"].diff()
    out["marginal_recall_loss_per_avoided_fire"] = (d_rec / d_fire).replace([np.inf, -np.inf], np.nan)
    base = (out.loc[0.5, "inscope_pair_recall"] - out.loc[0.9, "inscope_pair_recall"]) / \
           (out.loc[0.5, "fired_corpus_abstracts"] - out.loc[0.9, "fired_corpus_abstracts"])
    out["marginal_cost_vs_0.5-0.9_slope"] = out["marginal_recall_loss_per_avoided_fire"] / base

    with open(args.out_csv, "w") as f:
        f.write("# Cross-encoder gate analysis (R3.6): shipped arm, mean over seeds "
                f"{seeds}; see litdd/evaluation/crossencoder_gate_analysis.py.\n")
        f.write("# bench_* = frozen external benchmark (deployment-faithful paper recall vs fired "
                f"corpus abstracts of {args.n_corpus}); marginal columns compare consecutive rows.\n")
        out.round(5).to_csv(f)
    pd.set_option("display.width", 260)
    cols = ["test_precision", "test_recall", "test_f1", "inscope_pair_recall", "deploy_paper_recall",
            "fired_corpus_abstracts", "fire_pct", "bench_precision_lb", "bench_f1_lb", "bench_f05_lb",
            "marginal_recall_loss_per_avoided_fire", "marginal_cost_vs_0.5-0.9_slope"]
    print(out[cols].round(4).to_string())
    print(f"\nargmax bench_f1_lb = {out.bench_f1_lb.idxmax()}, argmax bench_f05_lb = "
          f"{out.bench_f05_lb.idxmax()}, argmax test_f1 = {out.test_f1.idxmax()}")
    print(f"[Info] wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
