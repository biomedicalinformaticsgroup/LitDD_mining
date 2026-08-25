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
  * drop retractions, retraction notices, errata and expressions of concern by MeSH
    publication type (see ``DEFAULT_EXCLUDE_PUBTYPES``);
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


# Retractions and corrections, excluded from the corpus by MeSH publication type.
# Counts are from the full 2026 download (45,056,462 records; see
# revision/kco/publication_types_2026.csv for all 85 types observed).
#
# Matched on the MeSH ID prefix, NOT the label: D016440 appears under two different labels
# in the same corpus ("Retraction Notice" 37,618 and the legacy "Retraction of Publication"
# 4), so label matching would miss records.
DEFAULT_EXCLUDE_PUBTYPES = (
    "D016441",  # Retracted Publication          36,775  the retracted article itself
    "D016440",  # Retraction Notice              37,622  the notice (covers both labels)
    "D016425",  # Published Erratum             242,873  correction notice, not a paper
    "D000075742",  # Expression of Concern        4,937
    "D016438",  # Duplicate Publication             930
)

# D016439 Corrected and Republished Article (1,776) is deliberately KEPT: the republished
# version is the corrected, valid science. Its superseded predecessor is removed instead, via
# the RepublishedIn RefType below -- which is what makes keeping the publication type safe.

# CommentsCorrections RefTypes to exclude. These are NOT MeSH and pubmed_parser does not
# expose them at all, so they must come from extract_corrections.py. RetractionIn is the more
# reliable marker for a retracted paper than the publication type, because it can be present
# on records whose publication_types was never updated to include D016441.
#
# The In/Of and In/For suffixes are opposites, so both directions are listed where both should
# go: the notice and the paper it concerns are each excluded.
DEFAULT_EXCLUDE_REFTYPES = (
    "RetractionIn",            # this paper WAS retracted
    "RetractionOf",            # this IS the retraction notice
    "ErratumIn",               # this paper has a published correction
    "ErratumFor",              # this IS the erratum notice
    "ExpressionOfConcernIn",   # concern raised about this paper
    "ExpressionOfConcernFor",  # this IS the concern notice
    "RepublishedIn",           # SUPERSEDED predecessor of a corrected republication;
                               # its RepublishedFrom counterpart (the corrected version)
                               # is deliberately NOT excluded.
)

# Deliberately NOT excluded: CommentIn/CommentOn (commentary, not correction),
# UpdateIn/UpdateOf, ReprintIn/ReprintOf, OriginalReportIn, SummaryForPatientsIn,
# AssociatedDataset, and D016426 Scientific Integrity Review (491).


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
    p.add_argument("--exclude_pubtypes", default=",".join(DEFAULT_EXCLUDE_PUBTYPES),
                   help="Comma-separated MeSH publication-type IDs to exclude, matched on the "
                        "`Dxxxxxxx:` prefix rather than the label (labels are not unique -- "
                        "D016440 appears as both 'Retraction Notice' and 'Retraction of "
                        "Publication'). Pass an empty string to disable.")
    p.add_argument("--exclude_reftypes", default=",".join(DEFAULT_EXCLUDE_REFTYPES),
                   help="Comma-separated CommentsCorrections RefTypes to exclude. Requires "
                        "--corrections_csv. Empty string disables.")
    p.add_argument("--corrections_csv", default=None,
                   help="CSV of pmid,reftype from extract_corrections.py. Defaults to "
                        "<download_dir>/corrections.csv if present. Without it, retracted "
                        "papers whose publication_types was never updated are NOT excluded.")
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
        select.append(
            pl.col("publication_types").cast(pl.Utf8).alias("publication_types")
            if "publication_types" in cols else pl.lit("").alias("publication_types")
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

    # Retractions / corrections, by MeSH ID prefix.
    excl = [t.strip() for t in args.exclude_pubtypes.split(",") if t.strip()]
    n_pubtype = 0
    if excl:
        pat = "|".join(f"{t}:" for t in excl)
        hit = pl.col("publication_types").fill_null("").str.contains(pat)
        pubtype_pmids = set(df.filter(hit)["pmid"].to_list())
        n_pubtype = len(pubtype_pmids)
        if pubtype_pmids:
            df = df.filter(~pl.col("pmid").is_in(list(pubtype_pmids)))

    # CommentsCorrections RefTypes -- the non-MeSH correction mechanism.
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
            print(f"[warn] no corrections file at {corr_path}; RefType exclusions skipped. "
                  f"Retracted papers whose publication_types was never updated will remain. "
                  f"Run extract_corrections.py first.")

    # Later shards supersede earlier ones for the same PMID.
    keep = (df.sort("shard_order", descending=True)
              .unique(subset=["pmid"], keep="first")
              .select(["pmid", "source_shard"])
              .sort("pmid"))

    keep.write_parquet(out_path)
    print(f"records scanned        : {total}")
    print(f"withdrawn (DeleteCitation): {len(deleted_pmids)} "
          f"(delete flag: {from_flag}, DeleteCitation file: {from_file})")
    print(f"duplicate occurrences  : "
          f"{total - len(deleted_pmids) - n_pubtype - n_reftype - keep.height}")
    print(f"excluded by pubtype    : {n_pubtype} ({','.join(excl) if excl else 'disabled'})")
    print(f"excluded by reftype    : {n_reftype} ({','.join(reftypes) if reftypes else 'disabled'})")
    print(f"unique PMIDs kept      : {keep.height}")
    print(f"manifest -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
