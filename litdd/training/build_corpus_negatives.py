#!/usr/bin/env python3
"""Sample corpus-representative negatives for the screen's training set.

WHY
---
The released screen fires on 19.46% of random 2026 PubMed records; the previously deployed one
fires on 2.48% (matching the historic 2.32%). Same rows, same fp32 path -- the only difference
is the checkpoint. The cause is training-set composition:

    ds_bert_train      11,201 rows, 25.5% positive  -> 2.48% corpus rate
    ds_hirecall_train  17,335 rows, 51.8% positive  -> 19.46% corpus rate
    ds_test             2,779 rows, 25.0% positive

The augmentation added ~6,100 positives and NO negatives, doubling training prevalence against
a deployment prevalence of roughly 1-2%. Worse, every negative in the set is a
`(tiab, g2p_lgmde)` pair drawn from gene-disease-relevant literature, so the model has never
seen an ordinary PubMed abstract -- a chemistry paper, an ecology survey, a drug trial -- and
has no reason to reject one. `random_csv` was only ever used for evaluation, never training.

ds_test cannot detect this: at 25% positive it is ~25x denser in positives than deployment, so
a model can look identical there (F1 0.91 vs 0.92) while behaving 8x differently at scale.

WHAT THIS DOES
--------------
Draws PMIDs at random from the FULL converted PubMed corpus -- every eligible record, not the
BERT-screened subset -- and treats them as negative unless plausibly positive. Excluded:
  * PMIDs cited in the G2P `publications` column of any supplied snapshot;
  * PMIDs in the external curated truth sets (premined / HPOA / ClinGen);
  * PMIDs already in the annotated train or test material (no leakage either way).

Residual contamination is bounded by the prevalence of uncurated DD gene-disease papers,
~1-2%, which is acceptable for negatives and is realistic label noise in any case.

SAMPLING IS DECADE-STRATIFIED BY DEFAULT. PubMed's own composition is heavily recency-weighted
(a uniform draw comes out ~43% from the 2020s and ~17% from the 1980s), and the released
model's false-positive rate varies strongly by era -- 10.66% in the 1980s rising to 25.55% in
the 2020s. Sampling uniformly would under-teach exactly the decades where behaviour differs
most. Equal-per-decade over-weights old literature relative to deployment, which is a
deliberate trade: we cannot train at the true ~1% prevalence anyway, so even coverage of the
error surface is worth more than matching the prior. Pass --uniform to match corpus
composition instead.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

csv.field_size_limit(10**9)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus_dir", default="data/pubmed_download_2026/parquet_download_files",
                   help="Converted PubMed parquet shards (the whole corpus, not BERT output)")
    p.add_argument("--g2p_csvs", nargs="+", default=["revision/G2P_DD_2026-06-24.csv"])
    p.add_argument("--truth_csvs", nargs="+",
                   default=["revision/external_recall/external_positives.csv",
                            "revision/external_recall/evagg_external_eval.csv"])
    p.add_argument("--exclude_csvs", nargs="+",
                   default=["revision/external_recall/annotated_tiab_augmented.csv"],
                   help="Train/test material whose PMIDs must never appear as new negatives")
    p.add_argument("--n", type=int, default=200000, help="Total negatives to draw")
    p.add_argument("--shards", type=int, default=300, help="Corpus shards to sample from")
    p.add_argument("--uniform", action="store_true",
                   help="Sample uniformly (matches corpus composition) instead of "
                        "equal-per-decade")
    p.add_argument("--min_year", type=int, default=1980)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="revision/external_recall/corpus_negatives.csv")
    return p.parse_args()


def pmids_from_csv(path: str) -> set[str]:
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


def main():
    a = parse_args()
    import random

    import polars as pl

    print("Building exclusion set:")
    excl: set[str] = set()
    for p in a.g2p_csvs:
        excl |= g2p_publication_pmids(p)
    for p in a.truth_csvs + a.exclude_csvs:
        excl |= pmids_from_csv(p)
    print(f"  TOTAL excluded: {len(excl):,} PMIDs")

    shards = sorted(glob.glob(os.path.join(a.corpus_dir, "*.parquet")))
    if not shards:
        raise SystemExit(f"no shards in {a.corpus_dir}")
    random.seed(a.seed)
    pick = random.sample(shards, min(a.shards, len(shards)))
    print(f"\nSampling from {len(pick)} of {len(shards)} corpus shards")

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
            # Report rather than silently under-fill: a thin decade changes the balance.
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
