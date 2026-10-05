#!/usr/bin/env python3
"""Resolve duplicate, withdrawn and retracted PubMed records after XML to parquet conversion.

Reads
    ``<download_dir>/parquet_download_files/*.parquet``  shards from ``pubmed_to_parquet.py``
    ``<download_dir>/deleted_pmids.txt``  withdrawn PMIDs from ``extract_deleted_pmids.py``
    ``<download_dir>/corrections.csv``    (pmid, reftype) links from ``extract_corrections.py``

Writes
    ``<download_dir>/pmid_keep.parquet``  manifest of (pmid, source_shard), one row per
    PMID kept, which ``bert_predict_vllm.py --keep_parquet`` filters against; the shards
    themselves are left unchanged.

MEDLINE updatefiles reissue records that already appear in the annual baseline and carry
``<DeleteCitation>`` entries for records PubMed has withdrawn. Unresolved, a duplicated PMID
passes through every stage twice and a withdrawn record stays in the corpus. The manifest
is computed as follows:

* drop every PMID listed in the withdrawn-PMID file or flagged ``delete`` by the parser
  (the parser emits no row for ``<DeleteCitation>`` entries, so the file is the working
  source; see ``extract_deleted_pmids.py``);
* drop records whose MeSH publication type marks them as a retracted article, a retraction
  notice, an erratum notice, an expression of concern or a duplicate publication
  (``DEFAULT_EXCLUDE_PUBTYPES``);
* drop records whose CommentsCorrections links mark them as retracted, as a retraction or
  concern notice, as the subject of an expression of concern, or as the earlier version
  of a republished article (``DEFAULT_EXCLUDE_REFTYPES``);
* for a PMID appearing in more than one shard, keep the occurrence from the
  highest-numbered file, since later distribution files supersede earlier ones.

Retracted articles and their notices are caught by publication type; the RefType pass adds
papers with an expression of concern and earlier versions of republished articles,
which no publication type expresses. Papers that merely have a published correction
(``ErratumIn``) are kept; the correction notices themselves are excluded by publication type.

Example
-------
    python -m litdd.pipeline.dedupe_pmids --download_dir data/pubmed_download
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from glob import glob

import polars as pl

logger = logging.getLogger("litdd.dedupe_pmids")

# MeSH publication types excluded from the corpus, matched on the descriptor id prefix
# rather than the label, because one id can appear under more than one label.
DEFAULT_EXCLUDE_PUBTYPES = (
    "D016441",     # Retracted Publication: the retracted article itself
    "D016440",     # Retraction Notice
    "D016425",     # Published Erratum: the correction notice
    "D000075742",  # Expression of Concern
    "D016438",     # Duplicate Publication
)
# D016439 Corrected and Republished Article is kept: the republished version is the valid
# record, and its earlier version is removed through the RepublishedIn link below.

# CommentsCorrections RefTypes excluded from the corpus. These are not MeSH terms and the
# parser does not expose them, so they come from extract_corrections.py. The ``In`` and
# ``Of``/``For`` suffixes point in opposite directions: ``RetractionIn`` is on the retracted
# paper, ``RetractionOf`` on the notice.
DEFAULT_EXCLUDE_REFTYPES = (
    "RetractionIn",              # this paper was retracted
    "RetractionOf",              # this record is the retraction notice
    "ExpressionOfConcernIn",     # a concern was raised about this paper
    "ExpressionOfConcernFor",    # this record is the concern notice
    "RepublishedIn",             # earlier version of a corrected republication
    "RetractedandRepublishedIn",  # earlier version that was retracted and republished
)
# Not excluded: RepublishedFrom and RetractedandRepublishedFrom (the corrected republications),
# ErratumIn (papers that have a correction), CommentIn/CommentOn, UpdateIn/UpdateOf,
# ReprintIn/ReprintOf, OriginalReportIn, SummaryForPatientsIn, AssociatedDataset and
# AssociatedPublication.


def shard_order_key(path: str) -> str:
    """Sort key placing later distribution files last (pubmed26n0001 < pubmed26n1274)."""
    return os.path.basename(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--download_dir", required=True,
                   help="Directory holding parquet_download_files/")
    p.add_argument("--out", default=None,
                   help="Manifest path (default <download_dir>/pmid_keep.parquet)")
    p.add_argument("--exclude_pubtypes", default=",".join(DEFAULT_EXCLUDE_PUBTYPES),
                   help="Comma-separated MeSH publication-type ids to exclude, matched on the "
                        "'Dxxxxxxx:' prefix rather than the label. Empty string disables.")
    p.add_argument("--exclude_reftypes", default=",".join(DEFAULT_EXCLUDE_REFTYPES),
                   help="Comma-separated CommentsCorrections RefTypes to exclude. Requires "
                        "--corrections_csv. Empty string disables.")
    p.add_argument("--corrections_csv", default=None,
                   help="CSV of pmid,reftype from extract_corrections.py. Defaults to "
                        "<download_dir>/corrections.csv if present.")
    p.add_argument("--deleted_pmids", default=None,
                   help="File of withdrawn PMIDs, one per line, from extract_deleted_pmids.py. "
                        "Defaults to <download_dir>/deleted_pmids.txt if present.")
    return p.parse_args(argv)


def scan_shards(shards: list[str]) -> pl.DataFrame:
    """Read pmid, publication_types and the delete flag from every shard, with the shard order."""
    frames = []
    for order, path in enumerate(shards):
        cols = pl.read_parquet_schema(path)
        select = [pl.col("pmid").cast(pl.Utf8), pl.lit(order).alias("shard_order"),
                  pl.lit(os.path.basename(path)).alias("source_shard")]
        # Baseline-only shards have no delete column; treat their rows as not deleted.
        select.append(
            pl.col("delete").cast(pl.Boolean).alias("delete") if "delete" in cols
            else pl.lit(False).alias("delete")
        )
        select.append(
            pl.col("publication_types").cast(pl.Utf8).alias("publication_types")
            if "publication_types" in cols else pl.lit("").alias("publication_types")
        )
        frames.append(pl.scan_parquet(path).select(select))
    return pl.concat(frames).collect()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    parquet_dir = os.path.join(args.download_dir, "parquet_download_files")
    out_path = args.out or os.path.join(args.download_dir, "pmid_keep.parquet")

    shards = sorted(glob(os.path.join(parquet_dir, "*.parquet")), key=shard_order_key)
    if not shards:
        logger.error("No parquet shards found in %s", parquet_dir)
        return 1
    logger.info("Scanning %d shard(s)", len(shards))

    df = scan_shards(shards)
    total = df.height

    # Withdrawn PMIDs: the parser's delete flag plus the file extracted from <DeleteCitation>.
    deleted_pmids = set(df.filter(pl.col("delete"))["pmid"].to_list())
    from_flag = len(deleted_pmids)
    deleted_path = args.deleted_pmids or os.path.join(args.download_dir, "deleted_pmids.txt")
    from_file = 0
    if os.path.exists(deleted_path):
        with open(deleted_path) as fh:
            extra = {ln.strip() for ln in fh if ln.strip()}
        from_file = len(extra)
        deleted_pmids |= extra
    else:
        logger.warning("no withdrawn-PMID file at %s; withdrawn papers are not excluded. "
                       "Run extract_deleted_pmids.py first.", deleted_path)
    df = df.filter(~pl.col("pmid").is_in(list(deleted_pmids)) if deleted_pmids else pl.lit(True))

    # Retractions and corrections by MeSH publication-type id prefix.
    excl = [t.strip() for t in args.exclude_pubtypes.split(",") if t.strip()]
    n_pubtype = 0
    if excl:
        pat = "|".join(f"{t}:" for t in excl)
        hit = pl.col("publication_types").fill_null("").str.contains(pat)
        pubtype_pmids = set(df.filter(hit)["pmid"].to_list())
        n_pubtype = len(pubtype_pmids)
        if pubtype_pmids:
            df = df.filter(~pl.col("pmid").is_in(list(pubtype_pmids)))

    # Retractions and corrections by CommentsCorrections RefType.
    reftypes = [t.strip() for t in args.exclude_reftypes.split(",") if t.strip()]
    n_reftype = 0
    if reftypes:
        corr_path = args.corrections_csv or os.path.join(args.download_dir, "corrections.csv")
        if os.path.exists(corr_path):
            wanted = set(reftypes)
            ref_pmids = set(
                pl.read_csv(corr_path, schema_overrides={"pmid": pl.Utf8})
                  .filter(pl.col("reftype").is_in(list(wanted)))["pmid"].to_list()
            )
            ref_pmids -= {None}
            n_reftype = len(ref_pmids)
            if ref_pmids:
                df = df.filter(~pl.col("pmid").is_in(list(ref_pmids)))
        else:
            logger.warning("no corrections file at %s; RefType exclusions skipped. "
                           "Run extract_corrections.py first.", corr_path)

    # Later shards supersede earlier ones for the same PMID.
    keep = (df.sort("shard_order", descending=True)
              .unique(subset=["pmid"], keep="first")
              .select(["pmid", "source_shard"])
              .sort("pmid"))

    keep.write_parquet(out_path)
    logger.info("records scanned        : %d", total)
    logger.info("withdrawn (DeleteCitation): %d (delete flag: %d, DeleteCitation file: %d)",
                len(deleted_pmids), from_flag, from_file)
    logger.info("duplicate occurrences  : %d",
                total - len(deleted_pmids) - n_pubtype - n_reftype - keep.height)
    logger.info("excluded by pubtype    : %d (%s)", n_pubtype, ",".join(excl) if excl else "disabled")
    logger.info("excluded by reftype    : %d (%s)", n_reftype, ",".join(reftypes) if reftypes else "disabled")
    logger.info("unique PMIDs kept      : %d", keep.height)
    logger.info("manifest -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
