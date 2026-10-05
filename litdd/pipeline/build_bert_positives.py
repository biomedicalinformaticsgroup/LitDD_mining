#!/usr/bin/env python3
"""Collect the screen's positive predictions into one parquet file.

Reads every ``*_bert_processed.parquet`` in ``--processed_dir`` (written by
``bert_predict_vllm.py``) and writes the rows with ``bert_predict == 1`` to ``--out_path``,
the input of ``gene_candidates.py``.

Usage:
    python -m litdd.pipeline.build_bert_positives \\
        --processed_dir data/bert_processed --out_path data/pubmed_bert_positive.parquet
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import polars as pl

logger = logging.getLogger("litdd.build_bert_positives")

PARQUET_COMPRESSION = "zstd"


def build_positive_parquet(processed_dir: str, out_path: str) -> int:
    """Write the positive rows of every processed shard to ``out_path``; return 1 when none exist."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    files = sorted(glob.glob(os.path.join(processed_dir, "*_bert_processed.parquet")))
    if not files:
        logger.error("No processed parquet files found in %s", processed_dir)
        return 1

    # Stream the shards through polars and keep only the positive class.
    lf = pl.scan_parquet(files).filter(pl.col("bert_predict") == 1)
    lf.sink_parquet(out_path, compression=PARQUET_COMPRESSION)

    try:
        n = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        logger.info("Saved %d rows with bert_predict == 1 to %s", n, out_path)
    except Exception:
        logger.info("Wrote positives to %s", out_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed_dir", type=str, default="data/bert_processed",
                    help="Directory of *_bert_processed.parquet shards")
    ap.add_argument("--out_path", type=str, default="pubmed_bert_positive.parquet",
                    help="Output parquet of positive rows")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    return build_positive_parquet(args.processed_dir, args.out_path)


if __name__ == "__main__":
    raise SystemExit(main())
