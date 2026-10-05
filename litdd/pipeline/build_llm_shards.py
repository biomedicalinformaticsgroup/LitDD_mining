"""Shard builder: turn the gene gate's candidates into the adjudication stage's inputs.

Input: ``candidates.parquet`` from ``gene_candidates.py`` with columns ``pmid``, ``tiab``,
``candidate_g2p_ids`` (list of G2P ids) and ``candidate_sources``; other columns are
allowed and ignored.

Output: ``<out_dir>/<stem>_shard{i}-of-{n}.parquet`` files, each with columns ``pmid``,
``tiab`` and ``candidates`` (list of strings, the ``candidate_g2p_ids`` in their stored
order). Rows whose candidate list is empty are dropped: they have nothing to adjudicate and
are reported as no-candidate rows by ``final_data_clean.py`` from ``candidates.parquet``.

Place in the pipeline: after ``gene_candidates.py`` and before ``llm_map.py``, which reads
the shard directory. The shard count controls the granularity of resumable units; row
striping across workers is done by ``llm_map.py`` itself.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import polars as pl

logger = logging.getLogger(__name__)

SHARD_COLUMNS = ["pmid", "tiab", "candidates"]


def shard_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Select ``pmid``, ``tiab`` and ``candidates`` from a candidates frame, dropping empty rows.

    ``candidates`` is ``candidate_g2p_ids`` cast to a list of strings. When the frame already
    has a ``candidates`` column and no ``candidate_g2p_ids``, it is used as is.
    """
    source = "candidate_g2p_ids" if "candidate_g2p_ids" in df.columns else "candidates"
    out = df.select(
        pl.col("pmid"),
        pl.col("tiab"),
        pl.col(source).cast(pl.List(pl.Utf8)).alias("candidates"),
    )
    return out.filter(pl.col("candidates").list.len() > 0)


def write_shards(df: pl.DataFrame, out_dir: str, num_shards: int | None = None,
                 rows_per_shard: int | None = None, stem: str = "corpus") -> list[str]:
    """Write the shard parquets for ``df`` and return their paths in shard order.

    Exactly one of ``num_shards`` and ``rows_per_shard`` must be given. Rows are split into
    contiguous blocks; the last shard takes the remainder. An input frame with no rows after
    dropping empty candidate lists produces no files.
    """
    if (num_shards is None) == (rows_per_shard is None):
        raise ValueError("give exactly one of num_shards and rows_per_shard")
    shard = shard_frame(df)
    n_rows = shard.height
    logger.info("%d of %d rows have candidates", n_rows, df.height)
    if n_rows == 0:
        logger.warning("no rows with candidates; no shards written")
        return []
    if rows_per_shard is not None:
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        n = math.ceil(n_rows / rows_per_shard)
    else:
        if num_shards is None or num_shards <= 0:
            raise ValueError("num_shards must be positive")
        n = min(num_shards, n_rows)
    per = math.ceil(n_rows / n)
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for i in range(n):
        part = shard.slice(i * per, per)
        if part.height == 0:
            break
        path = os.path.join(out_dir, f"{stem}_shard{i}-of-{n}.parquet")
        part.write_parquet(path)
        logger.info("wrote %s (%d rows, %d candidates)", path, part.height,
                    int(part["candidates"].list.len().sum()))
        paths.append(path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidates_parquet", required=True,
                   help="candidates.parquet from gene_candidates.py.")
    p.add_argument("--out_dir", required=True, help="Directory for the shard parquets.")
    split = p.add_mutually_exclusive_group()
    split.add_argument("--num_shards", type=int, default=None,
                       help="Number of shard files (default 1).")
    split.add_argument("--rows_per_shard", type=int, default=None,
                       help="Rows per shard file; the shard count follows from the row count.")
    p.add_argument("--stem", default="corpus", help="File-name stem of the shards.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args(argv)
    num_shards = args.num_shards
    if num_shards is None and args.rows_per_shard is None:
        num_shards = 1
    df = pl.read_parquet(args.candidates_parquet)
    logger.info("read %d rows from %s", df.height, args.candidates_parquet)
    paths = write_shards(df, args.out_dir, num_shards=num_shards,
                         rows_per_shard=args.rows_per_shard, stem=args.stem)
    logger.info("%d shard(s) written to %s", len(paths), args.out_dir)


if __name__ == "__main__":
    main()
