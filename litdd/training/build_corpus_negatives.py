#!/usr/bin/env python3
"""Sample corpus-representative negatives for the screen's training set.

Reads the converted PubMed parquet shards under ``--corpus_dir`` (the whole corpus, not the
screened subset) and draws English records published after ``--min_year`` with a non-empty
title or abstract. Excluded from the draw are PMIDs cited in the ``publications`` column of the
G2P snapshots (``--g2p_csvs``), PMIDs in the external truth sets (``--truth_csvs``) and PMIDs in
the annotated training and test material (``--exclude_csvs``). Every remaining record is
labelled 0; the residual contamination is bounded by the prevalence of uncurated gene-disease
papers in PubMed.

By default the ``--n`` records are drawn in equal numbers per publication decade, so the
negatives cover every era at the same density; ``--uniform`` draws in proportion to the corpus
instead. Decades with fewer records than the per-decade target are reported.

Writes ``--out`` as a CSV with ``pmid, tiab, g2p_lgmde, label, pubdate`` (``g2p_lgmde`` empty,
``label`` 0) and prints the exclusion counts and the per-decade tallies.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(10**9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_dir", default="data/pubmed_download_2026/parquet_download_files",
                   help="Converted PubMed parquet shards (the whole corpus, not screen output)")
    p.add_argument("--g2p_csvs", nargs="+", required=True,
                   help="G2P snapshots whose publications column lists PMIDs to exclude")
    p.add_argument("--truth_csvs", nargs="+", required=True,
                   help="External truth CSVs whose PMIDs are excluded")
    p.add_argument("--exclude_csvs", nargs="+", required=True,
                   help="Train/test material whose PMIDs must never appear as new negatives")
    p.add_argument("--n", type=int, default=200000, help="Total negatives to draw")
    p.add_argument("--shards", type=int, default=300, help="Corpus shards to sample from")
    p.add_argument("--uniform", action="store_true",
                   help="Sample uniformly (matches corpus composition) instead of "
                        "equal-per-decade")
    p.add_argument("--min_year", type=int, default=1980)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="Output CSV of negatives")
    return p.parse_args()


def pmids_from_csv(path: str) -> set[str]:
    """Numeric PMIDs from a CSV's ``pmid`` or ``PMID`` column; empty when the file or column is absent."""
    if not os.path.exists(path):
        print(f"  [skip] {path} (absent)")
        return set()
    out: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        col = next((c for c in ("pmid", "PMID") if c in (rd.fieldnames or [])), None)
        if col is None:
            print(f"  [skip] {path} (no pmid column)")
            return set()
        for r in rd:
            v = (r.get(col) or "").strip()
            if v.isdigit():
                out.add(v)
    print(f"  {len(out):>8,} PMIDs from {path}")
    return out


def g2p_publication_pmids(path: str) -> set[str]:
    """Numeric PMIDs from the ``publications`` column of a G2P CSV (``;`` or ``,`` separated)."""
    if not os.path.exists(path):
        print(f"  [skip] {path} (absent)")
        return set()
    out: set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for v in (r.get("publications") or "").replace(",", ";").split(";"):
                v = v.strip()
                if v.isdigit():
                    out.add(v)
    print(f"  {len(out):>8,} PMIDs from {path} (publications column)")
    return out


def main() -> int:
    a = parse_args()
    import random

    import polars as pl

    print("Building exclusion set:")
    excl: set[str] = set()
    for p in a.g2p_csvs:
        excl |= g2p_publication_pmids(p)
    for p in a.truth_csvs + a.exclude_csvs:
        excl |= pmids_from_csv(p)
    print(f"  total excluded: {len(excl):,} PMIDs")

    shards = sorted(glob.glob(os.path.join(a.corpus_dir, "*.parquet")))
    if not shards:
        raise SystemExit(f"no shards in {a.corpus_dir}")
    random.seed(a.seed)
    pick = random.sample(shards, min(a.shards, len(shards)))
    print(f"\nSampling from {len(pick)} of {len(shards)} corpus shards")

    # English records after min_year with a non-empty title or abstract.
    df = (pl.scan_parquet(pick)
            .filter((pl.col("languages") == "eng") & (pl.col("pubdate") > a.min_year))
            .select([pl.col("pmid").cast(pl.Utf8), pl.col("pubdate"),
                     (pl.col("title").fill_null("") + " " +
                      pl.col("abstract").fill_null("")).str.strip_chars().alias("tiab")])
            .filter(pl.col("tiab").str.len_chars() > 0)
            .collect())
    print(f"  eligible rows available : {df.height:,}")
    df = df.filter(~pl.col("pmid").is_in(list(excl))).unique(subset=["pmid"])
    df = df.with_columns((pl.col("pubdate") // 10 * 10).alias("decade"))
    print(f"  after exclusions        : {df.height:,}")
    print("  available by decade:")
    print(df.group_by("decade").len().sort("decade"))

    if a.uniform:
        out = df.sample(n=min(a.n, df.height), seed=a.seed, shuffle=True)
    else:
        # Equal draw per decade; decades below the target are reported.
        decades = sorted(df["decade"].unique().to_list())
        per = a.n // len(decades)
        parts = []
        short = []
        for d in decades:
            sub = df.filter(pl.col("decade") == d)
            take = min(per, sub.height)
            if take < per:
                short.append((d, sub.height))
            parts.append(sub.sample(n=take, seed=a.seed, shuffle=True))
        out = pl.concat(parts)
        print(f"\n  decade-stratified: target {per:,} per decade across {len(decades)}")
        for d, have in short:
            print(f"  [warn] {d}s has only {have:,} available, below the {per:,} target")

    out = (out.with_columns([pl.lit("").alias("g2p_lgmde"), pl.lit(0).alias("label")])
              .select(["pmid", "tiab", "g2p_lgmde", "label", "pubdate"]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.write_csv(a.out)
    print(f"\nwrote {a.out}: {out.height:,} negatives")
    print(out.select([(pl.col("pubdate") // 10 * 10).alias("decade")])
             .group_by("decade").len().sort("decade"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
