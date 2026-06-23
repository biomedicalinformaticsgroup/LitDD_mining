#!/usr/bin/env python3
"""Top-K accuracy benchmark for cross-encoder rerankers.

Compares the fine-tuned LitDD cross-encoder to off-the-shelf alternatives by
ranking every unique TIAB in ``ds_test`` against every unique candidate
``g2p_lgmde`` in the test set; for each positive (TIAB, G2P) row we record
whether the true G2P record appears in the top-K predicted candidates.

Output: ``cv_results.csv`` with columns ``model,top1,top2,top3,top4,top5``.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os

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
    p.add_argument("--data_dir", default="train_test")
    p.add_argument("--out_csv", default="benchmarking/cv_results.csv")
    p.add_argument("--models", nargs="+", default=None,
                   help="Model paths/IDs. Include the fine-tuned LitDD cross-encoder path here.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def load_existing(out_csv: str) -> set[str]:
    if not os.path.exists(out_csv):
        return set()
    with open(out_csv, newline="") as f:
        return {row["model"] for row in csv.DictReader(f) if row.get("model")}


def append_row(out_csv: str, row: dict) -> None:
    is_new = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "top1", "top2", "top3", "top4", "top5"])
        if is_new:
            w.writeheader()
        w.writerow(row)


def topk_for_model(ds_test, model_name: str, batch_size: int) -> dict:
    print(f"\n=== Evaluating cross-encoder: {model_name} ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = CrossEncoder(model_name, device=device, model_kwargs=model_kwargs)

    tiabs = sorted(set(ds_test["tiab"]))
    candidates = sorted(set(ds_test["g2p_lgmde"]))
    pairs = [(t, c) for t in tiabs for c in candidates]

    scores: list[float] = []
    for i in tqdm(range(0, len(pairs), batch_size), desc=f"score {model_name}"):
        s = model.predict(pairs[i:i + batch_size])
        scores.extend(float(x) for x in np.array(s).reshape(-1))

    n, m = len(tiabs), len(candidates)
    arr = np.array(scores, dtype=np.float32).reshape(n, m)
    order = np.argsort(arr, axis=1)[:, ::-1]
    text_to_ranked = {tiabs[i]: [candidates[j] for j in order[i]] for i in range(n)}

    labels = np.array(ds_test["label"]).astype(int)
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) == 0:
        metrics = {f"top{k}": 0.0 for k in range(1, 6)}
    else:
        hits = {k: 0 for k in range(1, 6)}
        for idx in pos_idx:
            t = ds_test[int(idx)]["tiab"]
            true_id = ds_test[int(idx)]["g2p_lgmde"]
            ranked = text_to_ranked.get(t, [])
            for k in range(1, 6):
                if true_id in ranked[:k]:
                    hits[k] += 1
        metrics = {f"top{k}": round(hits[k] / len(pos_idx), 6) for k in range(1, 6)}

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main() -> int:
    args = parse_args()
    ds_test = load_from_disk(os.path.join(args.data_dir, "ds_test"))
    existing = load_existing(args.out_csv) if args.skip_existing else set()
    models = args.models or DEFAULT_MODELS

    for name in models:
        if name in existing:
            print(f"[SKIP] {name} already in {args.out_csv}")
            continue
        try:
            metrics = topk_for_model(ds_test, name, args.batch_size)
            row = {"model": name, **metrics}
            append_row(args.out_csv, row)
            print("->", row)
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")
            append_row(args.out_csv, {"model": name, "top1": "", "top2": "", "top3": "", "top4": "", "top5": ""})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
