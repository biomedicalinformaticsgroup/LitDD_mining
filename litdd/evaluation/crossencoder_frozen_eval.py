#!/usr/bin/env python3
"""Cross-encoder evaluation on the frozen external benchmark — the screen's three-axis
release gate, in pair form (R3.4 / R3.5 / R3.6 / R2-P1).

For one checkpoint, scores three pair sets built by ``build_corpus_negative_pairs.py`` and
reports, per threshold in ``--thresholds``:

  1. **External recall** on the frozen leak-free positives: the truth pair
     (tiab, true G2P thread) scores >= t, at three denominators — all truth pairs, the
     gate-passing subset, and the in-scope subset (``scope_category == in_scope``) —
     matching the screen's 1,752 / 1,431 / 637 reporting.
  2. **Top-k coverage under the data-driven universe**: rank of the true entry among the
     abstract's own gene-gate candidates (what deployment actually ranks), k in 1/3/5.
  3. **Corpus fire rate**: fraction of the 87,600 frozen corpus abstracts (gated subset,
     since only those reach the cross-encoder) with >= 1 candidate pair scoring >= t —
     the deployment-scale false-fire proxy; silver labels, so precision is a lower bound.

Also scores ``ds_test_candidates`` for the same top-k-in-gated-universe number on the
annotated test set. Per-pair scores are dumped so thresholds can be re-read later without
re-scoring. fp32 by default; ``--dtype`` for labelled speed comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--arm_dir", default="revision/crossencoder/flat")
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9, 0.95])
    p.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out_dir", default="revision/crossencoder/frozen_eval")
    return p.parse_args()


def score(model, pairs, bs):
    if not pairs:
        return np.zeros(0)
    return np.asarray(model.predict(pairs, batch_size=bs)).reshape(-1)


def topk_in_universe(model, rows, thread_map, k_values, bs) -> dict:
    """rows: iterable of (tiab, true_g2p_id, candidate ids). Coverage of the truth entry
    within its own gated universe; rows with an empty universe count as misses."""
    pairs, owner = [], []
    for i, (tiab, _, cands) in enumerate(rows):
        for gid in cands:
            if gid in thread_map:
                pairs.append((tiab, thread_map[gid]))
                owner.append((i, gid))
    s = score(model, pairs, bs)
    ranked = defaultdict(list)
    for (i, gid), sc in zip(owner, s):
        ranked[i].append((float(sc), gid))
    hits = {k: 0 for k in k_values}
    n = 0
    for i, (_, true_gid, _) in enumerate(rows):
        n += 1
        order = [g for _, g in sorted(ranked.get(i, []), reverse=True)]
        for k in k_values:
            if true_gid in order[:k]:
                hits[k] += 1
    return {f"top{k}_gated_universe": round(hits[k] / n, 4) if n else 0.0 for k in k_values}


def main() -> int:
    args = parse_args()
    from sentence_transformers import CrossEncoder

    dt = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = CrossEncoder(args.model, model_kwargs={"torch_dtype": dt})
    with open(os.path.join(args.arm_dir, "corpus.json")) as f:
        thread_map = {k: v for k, v in json.load(f).items() if not k.startswith("__")}
    os.makedirs(args.out_dir, exist_ok=True)
    out: dict = {"run_name": args.run_name, "model": args.model, "dtype": args.dtype}

    # 1. external recall on truth pairs, three denominators
    pos = load_from_disk(os.path.join(args.arm_dir, "ds_frozen_pos"))
    s_pos = score(model, list(zip(pos["tiab"], pos["g2p_lgmde"])), args.batch_size)
    gated = np.array([len(c) > 0 for c in pos["candidate_g2p_ids"]])
    in_scope = np.array([sc == "in_scope" for sc in pos["scope_category"]])
    with open(os.path.join(args.out_dir, f"{args.run_name}__frozen_pos_scores.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "g2p_id", "scope_category", "gated", "score"])
        for pm, g, sc, gt, s in zip(pos["pmid"], pos["g2p_id"], pos["scope_category"], gated, s_pos):
            w.writerow([pm, g, sc, int(gt), f"{s:.6f}"])
    for t in args.thresholds:
        hit = s_pos >= t
        out[f"recall_all@{t}"] = round(float(hit.mean()), 4)
        out[f"recall_gated@{t}"] = round(float(hit[gated].mean()), 4) if gated.any() else None
        out[f"recall_inscope@{t}"] = round(float(hit[in_scope].mean()), 4) if in_scope.any() else None
    out["n_pos_pairs"], out["n_pos_gated"], out["n_pos_inscope"] = \
        int(len(s_pos)), int(gated.sum()), int(in_scope.sum())

    # 2. top-k within the gated universe (frozen positives, and annotated test positives)
    rows = list(zip(pos["tiab"], pos["g2p_id"], pos["candidate_g2p_ids"]))
    out.update({f"frozen_{k}": v for k, v in
                topk_in_universe(model, rows, thread_map, args.k_values, args.batch_size).items()})
    test = load_from_disk(os.path.join(args.arm_dir, "ds_test_candidates"))
    trows = [(r["tiab"], r["g2p_lgmde"].split(" - ", 1)[0].strip(), r["candidate_g2p_ids"])
             for r in test if r["label"] == 1]
    out.update({f"dstest_{k}": v for k, v in
                topk_in_universe(model, trows, thread_map, args.k_values, args.batch_size).items()})

    # 3. corpus fire rate on gated frozen negatives (silver)
    neg = load_from_disk(os.path.join(args.arm_dir, "ds_frozen_neg_pairs"))
    s_neg = score(model, list(zip(neg["tiab"], neg["g2p_lgmde"])), args.batch_size)
    by_pmid = defaultdict(float)
    for pm, s in zip(neg["pmid"], s_neg):
        by_pmid[pm] = max(by_pmid[pm], float(s))
    n_gated_abstracts = len(by_pmid)
    with open(os.path.join(args.out_dir, f"{args.run_name}__frozen_neg_max_scores.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["pmid", "max_score"])
        for pm, s in by_pmid.items():
            w.writerow([pm, f"{s:.6f}"])
    mx = np.array(list(by_pmid.values()))
    for t in args.thresholds:
        fired = int((mx >= t).sum())
        out[f"fire_gated_abstracts@{t}"] = fired
        out[f"fire_rate_of_87600@{t}"] = round(100 * fired / 87600, 3)
    out["n_neg_gated_abstracts"], out["n_neg_pairs"] = n_gated_abstracts, int(len(s_neg))

    with open(os.path.join(args.out_dir, f"{args.run_name}__summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
