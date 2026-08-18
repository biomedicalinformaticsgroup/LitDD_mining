#!/usr/bin/env python3
"""Mine hard-negative (TIAB, G2P_LGMDE) pairs for cross-encoder training.

Uses a sentence-transformer to embed every TIAB and every candidate G2P
record, then picks the *most similar but not the gold-positive* G2P
records as hard negatives. Output is a HuggingFace ``save_to_disk``
directory consumable by ``crossencode_finetune.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

G2P_LGMDE_COLS = [
    "g2p id", "gene symbol", "gene mim", "hgnc id", "previous gene symbols",
    "disease name", "disease mim", "disease MONDO", "allelic requirement",
    "cross cutting modifier", "confidence", "inferred variant consequence",
    "variant types", "molecular mechanism", "molecular mechanism categorisation",
]


def build_g2p_lgmde(g2p: pd.DataFrame) -> pd.Series:
    """Concatenate the canonical G2P columns into the LGMDE join string."""
    cols = [c for c in G2P_LGMDE_COLS if c in g2p.columns]
    return g2p[cols].astype(str).agg(" - ".join, axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ds_cross_dir", default="litdd/training/ds_cross_train",
                   help="HuggingFace save_to_disk directory containing the (tiab, g2p_lgmde, label) train pairs.")
    p.add_argument("--g2p_csv", required=True,
                   help="G2P DD CSV (used to build the candidate-negatives corpus).")
    p.add_argument("--out_dir", default="litdd/training/hard_negatives_dataset")
    p.add_argument("--embed_model", default="abhinand/MedEmbed-large-v0.1")
    p.add_argument("--num_negatives", type=int, default=5)
    p.add_argument("--range_min", type=int, default=5)
    p.add_argument("--range_max", type=int, default=50)
    p.add_argument("--max_score", type=float, default=0.95)
    p.add_argument("--relative_margin", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=128)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds_cross = load_from_disk(args.ds_cross_dir)

    g2p = pd.read_csv(args.g2p_csv)
    if "g2p_lgmde" not in g2p.columns:
        g2p["g2p_lgmde"] = build_g2p_lgmde(g2p)

    print(f"[Info] {len(g2p)} G2P candidates, {len(ds_cross)} train pairs")

    embedding_model = SentenceTransformer(args.embed_model)
    hard_negatives_dataset = mine_hard_negatives(
        dataset=ds_cross,
        model=embedding_model,
        anchor_column_name="tiab",
        positive_column_name="g2p_lgmde",
        corpus=list(g2p["g2p_lgmde"]),
        range_min=args.range_min,
        range_max=args.range_max,
        max_score=args.max_score,
        relative_margin=args.relative_margin,
        num_negatives=args.num_negatives,
        sampling_strategy="top",
        batch_size=args.batch_size,
        output_format="labeled-pair",
        use_faiss=False,
    )
    Path(args.out_dir).parent.mkdir(parents=True, exist_ok=True)
    hard_negatives_dataset.save_to_disk(args.out_dir)
    print(f"[Info] saved hard-negatives dataset → {args.out_dir} ({len(hard_negatives_dataset)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
