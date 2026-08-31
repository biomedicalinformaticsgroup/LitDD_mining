#!/usr/bin/env python3
"""Top-K accuracy benchmark for cross-encoder rerankers (Reviewer 3 R3.5).

Ranks every unique TIAB in ``ds_test`` against a candidate pool and records, for each
positive (TIAB, G2P) row, whether the true G2P record appears in the top-K.

Two candidate pools, and the difference between them is the point:

* **test pool** (default) — only the ``g2p_lgmde`` values present in ``ds_test``. Cheap,
  but it flatters the score, because deployment never gets a shortlist this small.
* **full panel** (``--g2p_csv``) — every entry in the DDG2P panel (~2,800), which is what
  ``litdd/pipeline/crossencode.py`` actually ranks against at inference time. This is the
  number that justifies a deployment ``--top_k``.

Report both: the gap quantifies how much the test-set pool overstates coverage.

Output: ``--out_csv`` with columns ``model,pool,n_candidates,top<k>…`` for each requested k.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys

import numpy as np
import torch
from datasets import load_from_disk
from sentence_transformers import CrossEncoder
from tqdm import tqdm

DEFAULT_MODELS = [
    "ncbi/MedCPT-Cross-Encoder",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--test_subdir", default="ds_test",
                   help="Test dataset directory under --data_dir (e.g. ds_cross_test).")
    p.add_argument("--out_csv", default="results/cv_results.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model paths/IDs. Include the fine-tuned LitDD cross-encoder path here.")
    p.add_argument("--k_values", type=int, nargs="+", default=[1, 3, 5, 7, 10],
                   help="k values to report (default: 1 3 5 7 10)")
    p.add_argument("--g2p_csv", default=None,
                   help="DDG2P CSV. When given, ALSO score against the full panel — the "
                        "deployment-faithful pool. Costs len(tiabs) x len(panel) pairs.")
    p.add_argument("--corpus_json", default=None,
                   help="corpus.json from build_crossencoder_dataset.py — the full panel in "
                        "a specific thread-variant rendering. Takes precedence over --g2p_csv "
                        "for the full-panel pool (a variant-trained model must be benchmarked "
                        "against its own rendering).")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                   help="Model dtype for scoring. fp32 is the reporting standard; pass "
                        "fp16/bf16 only for explicitly-labelled speed comparisons.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def load_existing(out_csv: str) -> set[tuple[str, str]]:
    """Return (model, pool) pairs already scored, so --skip_existing is pool-aware."""
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {(row["model"], row.get("pool", "test")) for row in csv.DictReader(f)
                if row.get("model")}


def append_row(out_csv: str, row: dict, k_values: list[int]) -> None:
    is_new = not os.path.exists(out_csv)
    fieldnames = ["model", "pool", "n_candidates"] + [f"top{k}" for k in k_values]
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            w.writeheader()
        w.writerow(row)


def load_full_panel(g2p_csv: str) -> list[str]:
    """The complete DDG2P candidate pool, built exactly as inference builds it.

    Reuses ``build_g2p_lgmde_list`` from the deployment path so the benchmark cannot drift
    from the strings ``crossencode.py`` actually scores.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    from litdd.pipeline.crossencode import build_g2p_lgmde_list
    return list(build_g2p_lgmde_list(g2p_csv))


def topk_for_model(ds_test, model_name: str, batch_size: int,
                   k_values: list[int], candidates: list[str] | None = None,
                   dtype: str = "fp32") -> dict:
    """Top-k coverage of the true G2P record for one model over one candidate pool.

    ``candidates`` defaults to the distinct ``g2p_lgmde`` values in ds_test; pass the full
    panel for the deployment-faithful number.
    """
    print(f"\n=== Evaluating cross-encoder: {model_name} ({dtype}) ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16}[dtype]
    model = CrossEncoder(model_name, device=device,
                         model_kwargs={"torch_dtype": torch_dtype})

    tiabs = sorted(set(ds_test["tiab"]))
    if candidates is None:
        candidates = sorted(set(ds_test["g2p_lgmde"]))
    pairs = [(t, c) for t in tiabs for c in candidates]
    print(f"    {len(tiabs)} TIAB x {len(candidates)} candidates = {len(pairs):,} pairs")

    scores: list[float] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"score {model_name}"):
        s = model.predict(pairs[i:i + batch_size])
        scores.extend(float(x) for x in np.array(s).reshape(-1))

    n, m = len(tiabs), len(candidates)
    arr = np.array(scores, dtype=np.float32).reshape(n, m)
    max_k = max(k_values)
    # Only the top max_k matter, so argpartition beats a full argsort once the panel is
    # ~2,800 wide rather than the few dozen candidates a test-set pool holds.
    top_idx = np.argpartition(-arr, kth=min(max_k, m - 1), axis=1)[:, :max_k]
    ordered = np.take_along_axis(top_idx, np.argsort(-np.take_along_axis(arr, top_idx, 1), 1), 1)
    text_to_ranked = {tiabs[i]: [candidates[j] for j in ordered[i]] for i in range(n)}

    labels = np.array(ds_test["label"]).astype(int)
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) == 0:
        metrics = {f"top{k}": 0.0 for k in k_values}
    else:
        hits = dict.fromkeys(k_values, 0)
        for idx in pos_idx:
            row = ds_test[int(idx)]
            ranked = text_to_ranked.get(row["tiab"], [])
            for k in k_values:
                if row["g2p_lgmde"] in ranked[:k]:
                    hits[k] += 1
        metrics = {f"top{k}": round(hits[k] / len(pos_idx), 6) for k in k_values}

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metrics["n_candidates"] = len(candidates)
    return metrics


def main() -> int:
    args = parse_args()
    ds_test = load_from_disk(os.path.join(args.data_dir, args.test_subdir))
    existing = load_existing(args.out_csv) if args.skip_existing else set()
    models = args.models or DEFAULT_MODELS

    pools: list[tuple[str, list[str] | None]] = [("test", None)]
    if args.corpus_json:
        import json
        with open(args.corpus_json) as f:
            pools.append(("full_panel", sorted(set(str(v) for v in json.load(f).values()))))
    elif args.g2p_csv:
        pools.append(("full_panel", load_full_panel(args.g2p_csv)))

    for pool_name, candidates in pools:
        for name in models:
            if (name, pool_name) in existing:
                print(f"[SKIP] {name} / {pool_name} already in {args.out_csv}")
                continue
            try:
                metrics = topk_for_model(ds_test, name, args.batch_size,
                                         args.k_values, candidates, dtype=args.dtype)
                row = {"model": name, "pool": pool_name, **metrics}
                append_row(args.out_csv, row, args.k_values)
                print("->", row)
            except Exception as e:
                print(f"[ERROR] {name} / {pool_name} failed: {e}")
                append_row(args.out_csv,
                           {"model": name, "pool": pool_name, "n_candidates": "",
                            **{f"top{k}": "" for k in args.k_values}},
                           args.k_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
