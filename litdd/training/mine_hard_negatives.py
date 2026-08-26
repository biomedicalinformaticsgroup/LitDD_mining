#!/usr/bin/env python3
"""Mine hard-negative (TIAB, G2P_LGMDE) pairs for cross-encoder training.

Uses a sentence-transformer to embed every TIAB and every candidate G2P
record, then picks the *most similar but not the gold-positive* G2P
records as hard negatives. Output is a HuggingFace ``save_to_disk``
directory consumable by ``crossencode_finetune.py``: labeled pairs
``(tiab, g2p_lgmde, label)`` — the input positives, the mined negatives,
and (by default) the human-annotated negative pairs from the input set.

Only ``label == 1`` rows are offered to the miner as anchors. Mining over
the full pair set would hand the annotated *negative* pairs to
``mine_hard_negatives`` as if they were positives, which both mislabels
those pairs in the output and steers the "not the gold positive" exclusion
wrong. The annotated negatives instead re-enter the output as what they
are: labeled negatives — gene-sharing near-misses a clinician rejected,
which is exactly the candidate distribution the deployed cross-encoder
scores now that the gene filter runs ahead of it. Disable with
``--no_annotated_negatives`` to reproduce the original mined-only recipe.

The candidate corpus should be the same rendering the pairs use — pass the
``corpus.json`` written by ``build_crossencoder_dataset.py`` for the arm
being trained (``--corpus_json``), or fall back to the flat rendering of a
raw G2P export (``--g2p_csv``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives


def load_corpus(corpus_json: str | None, g2p_csv: str | None) -> list[str]:
    if corpus_json:
        with open(corpus_json) as f:
            return sorted(set(str(v) for v in json.load(f).values()))
    if g2p_csv:
        from litdd.threads import build_lgmde_list
        return build_lgmde_list(g2p_csv)
    raise SystemExit("one of --corpus_json / --g2p_csv is required")


def mine_labeled_pairs(ds_cross: Dataset, corpus: list[str], embed_model: str,
                       include_annotated_negatives: bool = True,
                       num_negatives: int = 5, range_min: int = 5, range_max: int = 50,
                       max_score: float = 0.95, relative_margin: float = 0.01,
                       batch_size: int = 128,
                       embedder: SentenceTransformer | None = None) -> Dataset:
    """Positives -> mined negatives (+ annotated negatives) as one labeled-pair set."""
    if "label" not in ds_cross.column_names:
        raise SystemExit("input dataset has no 'label' column; refusing to treat "
                         "every row as a positive.")
    ds_pos = ds_cross.filter(lambda r: r["label"] == 1)
    ds_neg = ds_cross.filter(lambda r: r["label"] == 0)
    print(f"[Info] anchors: {len(ds_pos)} positives "
          f"({len(ds_neg)} annotated negatives "
          f"{'appended' if include_annotated_negatives else 'EXCLUDED'})")

    if embedder is None:
        embedder = SentenceTransformer(embed_model)
    mined = mine_hard_negatives(
        dataset=ds_pos.select_columns(["tiab", "g2p_lgmde"]),
        model=embedder,
        anchor_column_name="tiab",
        positive_column_name="g2p_lgmde",
        corpus=corpus,
        range_min=range_min,
        range_max=range_max,
        max_score=max_score,
        relative_margin=relative_margin,
        num_negatives=num_negatives,
        sampling_strategy="top",
        batch_size=batch_size,
        output_format="labeled-pair",
        use_faiss=False,
    )
    if not include_annotated_negatives or len(ds_neg) == 0:
        return mined
    annotated = ds_neg.select_columns(["tiab", "g2p_lgmde", "label"]).cast(mined.features)
    return concatenate_datasets([mined, annotated])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ds_cross_dir", default="litdd/training/ds_cross_train",
                   help="HuggingFace save_to_disk directory with (tiab, g2p_lgmde, label) pairs.")
    p.add_argument("--corpus_json", default=None,
                   help="corpus.json from build_crossencoder_dataset.py (the arm's rendering).")
    p.add_argument("--g2p_csv", default=None,
                   help="Raw G2P export; flat-rendered as the corpus if no --corpus_json.")
    p.add_argument("--out_dir", default="litdd/training/hard_negatives_dataset")
    p.add_argument("--embed_model", default="abhinand/MedEmbed-large-v0.1")
    p.add_argument("--no_annotated_negatives", dest="include_annotated_negatives",
                   action="store_false",
                   help="Mined-only output (the original recipe); annotated label-0 pairs "
                        "are dropped from training.")
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
    corpus = load_corpus(args.corpus_json, args.g2p_csv)
    print(f"[Info] {len(corpus)} candidate threads, {len(ds_cross)} input pairs")

    out = mine_labeled_pairs(
        ds_cross, corpus, args.embed_model,
        include_annotated_negatives=args.include_annotated_negatives,
        num_negatives=args.num_negatives, range_min=args.range_min,
        range_max=args.range_max, max_score=args.max_score,
        relative_margin=args.relative_margin, batch_size=args.batch_size,
    )
    Path(args.out_dir).parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(args.out_dir)
    n_pos = sum(1 for x in out["label"] if x == 1)
    print(f"[Info] saved labeled pairs -> {args.out_dir} "
          f"({len(out)} rows: {n_pos} pos / {len(out) - n_pos} neg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
