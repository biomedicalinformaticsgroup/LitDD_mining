#!/usr/bin/env python3
"""Resolve duplicate and withdrawn PubMed records after XML -> parquet conversion.

Two facts about the MEDLINE distribution make this necessary, and neither was handled
before: updatefiles **reissue** records that already appear in the annual baseline (a
revised abstract, a corrected date), and they carry ``DeleteCitation`` entries for records
PubMed has withdrawn.

.. warning::
   ``pubmed_parser`` exposes a ``delete`` field but **it never fires**: measured on the full
   2026 corpus it is True for 0 of 45,056,462 records, across both baseline and updatefiles,
   even though the raw updatefile XML plainly contains ``<DeleteCitation>`` blocks.
   ``<DeleteCitation>`` is a sibling of ``<PubmedArticle>`` holding bare ``<PMID>`` elements
   with no article body, so a parser walking article records never emits a row for them.
   Withdrawn PMIDs must therefore be supplied via ``--deleted_pmids``, produced by
   ``extract_deleted_pmids.py``, which reads the raw XML directly. The ``delete`` column is
   still honoured if it ever starts working, but it cannot be relied on alone.

Left alone, both propagate: a duplicated PMID is screened twice, cross-encoded twice and
adjudicated twice, so it can contribute two identical ``(PMID, G2P_ID)`` rows to the final
map and inflate every count in the cascade funnel; a withdrawn record stays in the corpus.

Resolution, applied over the whole converted corpus:
  * drop every PMID marked ``delete``;
  * for PMIDs appearing in more than one shard, keep the occurrence from the
    highest-numbered file — the distribution is ordered so later files supersede earlier
    ones, which is what makes a reissued record a correction rather than a duplicate.

Writes a ``pmid_keep.parquet`` manifest of (pmid, source_shard) that ``bert_predict.py``
filters against, so the parquet shards themselves stay untouched and re-runnable.

Example
-------
    python litdd/pipeline/dedupe_pmids.py --download_dir data/pubmed_download_2026
"""
from __future__ import annotations

import argparse
import os
from glob import glob

import polars as pl


def shard_order_key(path: str) -> str:
    """Sort key placing later distribution files last (pubmed26n0001 < pubmed26n1274)."""
    return os.path.basename(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Directory holding parquet_download_files/")
    p.add_argument("--out", default=None,
                   help="Manifest path (default <download_dir>/pmid_keep.parquet)")
    p.add_argument("--deleted_pmids", default=None,
                   help="File of withdrawn PMIDs, one per line, from extract_deleted_pmids.py. "
                        "Required in practice: pubmed_parser's `delete` flag never fires "
                        "(0/45,056,462 on the 2026 corpus) because <DeleteCitation> carries no "
                        "article body. Defaults to <download_dir>/deleted_pmids.txt if present.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    parquet_dir = os.path.join(args.download_dir, "parquet_download_files")
    out_path = args.out or os.path.join(args.download_dir, "pmid_keep.parquet")

    shards = sorted(glob(os.path.join(parquet_dir, "*.parquet")), key=shard_order_key)
    if not shards:
        raise SystemExit(f"No parquet shards found in {parquet_dir}")
    print(f"Scanning {len(shards)} shard(s)")

    frames = []
    for order, path in enumerate(shards):
        cols = pl.read_parquet_schema(path)
        select = [pl.col("pmid").cast(pl.Utf8), pl.lit(order).alias("shard_order"),
                  pl.lit(os.path.basename(path)).alias("source_shard")]
        # `delete` is absent from baseline-only shards; treat those as not-deleted.
        select.append(
            pl.col("delete").cast(pl.Boolean).alias("delete") if "delete" in cols
            else pl.lit(False).alias("delete")
        )
        frames.append(pl.scan_parquet(path).select(select))

    df = pl.concat(frames).collect()
    total = df.height

    deleted_pmids = set(df.filter(pl.col("delete"))["pmid"].to_list())
    from_flag = len(deleted_pmids)

    # The real source of withdrawals: PMIDs parsed straight out of <DeleteCitation>.
    deleted_path = args.deleted_pmids or os.path.join(args.download_dir, "deleted_pmids.txt")
    from_file = 0
    if os.path.exists(deleted_path):
        with open(deleted_path) as fh:
            extra = {ln.strip() for ln in fh if ln.strip()}
        from_file = len(extra)
        deleted_pmids |= extra
    else:
        print(f"[warn] no withdrawn-PMID file at {deleted_path}; withdrawn papers will NOT be "
              f"excluded (pubmed_parser's `delete` flag alone finds none). "
              f"Run extract_deleted_pmids.py first.")
    df = df.filter(~pl.col("pmid").is_in(list(deleted_pmids)) if deleted_pmids else pl.lit(True))

    # Later shards supersede earlier ones for the same PMID.
    keep = (df.sort("shard_order", descending=True)
              .unique(subset=["pmid"], keep="first")
              .select(["pmid", "source_shard"])
              .sort("pmid"))

    keep.write_parquet(out_path)
    print(f"records scanned        : {total}")
    print(f"withdrawn (DeleteCitation): {len(deleted_pmids)} "
          f"(delete flag: {from_flag}, DeleteCitation file: {from_file})")
    print(f"duplicate occurrences  : {total - len(deleted_pmids) - keep.height}")
    print(f"unique PMIDs kept      : {keep.height}")
    print(f"manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
