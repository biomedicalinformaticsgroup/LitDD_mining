#!/usr/bin/env python3
"""Restrict each BERT-positive abstract to the G2P entries whose gene it mentions.

Runs between ``build_bert_positives.py`` and ``crossencode.py``. Previously the gene-mention
check was the *last* gate, applied in ``final_data_clean.py`` after the cross-encoder had
scored every abstract against all ~2,861 G2P entries. Moving it here collapses the candidate
set to the handful an abstract's genes support -- G2P holds 2,861 entries across 2,552 genes at
a median of 1 entry each -- which removes ~1,300x of the pairwise scoring and makes the number
of candidates shown to the LLM data-driven rather than a fixed top-5.

Because this is now a gate, its recall bounds the whole pipeline. Measured before adopting it
(``litdd/evaluation/gene_filter_recall.py``, results in
``revision/external_recall/gene_filter_summary.csv``): on the independent curated sets it
retains 98.8% of true (paper, gene) pairs with PubTator alone and 99.3% with the HGNC
descriptive-name complement; on the labelled annotation set it costs 3.3% recall and raises
precision from 0.254 to 0.299.

Provenance is recorded per row (``symbol_match`` / ``name_match``) so the precision audit can
report the two sources separately and either can be down-weighted later without re-running.

Example
-------
    python litdd/pipeline/gene_candidates.py \\
        --input_parquet data/pubmed_bert_positive.parquet \\
        --g2p_csv data/G2P_DD_2026-06-24.csv \\
        --gene2pubtator data/gene2pubtator3 \\
        --gene_info data/human_gene_info.gz \\
        --hgnc data/reference/hgnc_complete_set.txt \\
        --out_parquet data/bert_positive_candidates.parquet
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from litdd.genes import GeneNameMatcher, load_gene_info, load_pubtator_genes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_parquet", required=True, help="BERT-positive parquet (pmid, tiab)")
    p.add_argument("--g2p_csv", required=True)
    p.add_argument("--gene2pubtator", required=True, help="gene2pubtator3 bulk file (.gz or TSV)")
    p.add_argument("--gene_info", required=True, help="NCBI gene_info.gz (human filter)")
    p.add_argument("--hgnc", default=None,
                   help="hgnc_complete_set.txt for descriptive-name matching. Recommended: "
                        "recovers ~42%% of residual external-truth misses.")
    p.add_argument("--out_parquet", required=True)
    p.add_argument("--family_stems", action="store_true",
                   help="Also match enzyme-family stems, so \"the two human arginase genes\" "
                        "resolves to ARG1/ARG2 even without a numeral. Off by default: full "
                        "HGNC names only. Disease-named genes (\"Bardet-Biedl syndrome 1\") "
                        "are blocklisted from forming stems either way, so enabling this "
                        "cannot make a syndrome mention match its whole gene family.")
    p.add_argument("--symbol_fallback", action="store_true",
                   help="For abstracts where PubTator3 has NO human gene annotation at all "
                        "(coverage gap: old or title-only records), match G2P gene symbols "
                        "verbatim in the text (case-sensitive, word-bounded, >=3 characters, "
                        "minus SYMBOL_FALLBACK_BLOCKLIST). Never applied when PubTator "
                        "annotated the abstract, so it cannot add ambiguity there. Measured on "
                        "the annotated test split: recovers 9 of 9 curated abstracts PubTator "
                        "left unannotated (DMD, ITPR1, NEXMIF, SCN8A, CPLANE1).")
    p.add_argument("--keep_unmatched", action="store_true",
                   help="Keep rows with no detected gene and give them the FULL panel as "
                        "candidates, instead of dropping them. Measured as unnecessary "
                        "(the gate retains 98.8%% of external-truth pairs), but retained so "
                        "the hard-gate/hybrid comparison can be re-run.")
    return p.parse_args()


# Official G2P symbols that are also English words or clinical abbreviations. Only consulted
# by --symbol_fallback, i.e. for abstracts PubTator did not annotate at all.
SYMBOL_FALLBACK_BLOCKLIST = frozenset({
    "CAT", "SET", "MAX", "WAS", "ACHE", "STAR", "REST", "MARS", "AIP", "MET", "ATM", "ARC",
    "BAD", "BID", "CAD", "CAP", "COIL", "DIP", "FAT", "FLOT", "GAN", "HIP", "IMPACT", "KIT",
    "LAMP", "LARGE", "MICE", "NODAL", "OCT", "PIGS", "RAN", "SHE", "SON", "SPAG", "TANK",
    "TAT", "TUB", "WARS", "APEX", "CRISP", "MASS", "MICAL", "PALM", "PIGN", "SCAN", "SHOX",
    "SLIT", "TRIP", "WISP", "ARMS", "ATP", "DNA", "RNA", "EEG", "MRI", "CNS", "IQ", "ASD",
    "ADHD", "PCR", "CGH", "SNP", "CNV", "NGS", "WES", "WGS", "HPO", "MIM", "OMIM",
})


def find_symbols_verbatim(text: str, symbols: set[str]) -> set[str]:
    """Panel symbols appearing verbatim (case-sensitive, word-bounded) in ``text``."""
    if not text:
        return set()
    # hyphenated symbols (NKX2-1, HLA-DRB1) are kept whole; a hyphen followed by lowercase
    # ("ITPR1-related") ends the token so the bare symbol still matches
    tokens = set(re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![A-Za-z0-9])", text))
    return {t for t in tokens if len(t) >= 3 and t in symbols and t not in SYMBOL_FALLBACK_BLOCKLIST}


def load_gene_to_g2p(path: str) -> dict[str, list[str]]:
    """gene symbol (and previous symbols) -> [g2p_id, ...]."""
    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = (row.get("g2p id") or "").strip()
            if not gid:
                continue
            syms = [(row.get("gene symbol") or "").strip()]
            syms += [s.strip() for s in (row.get("previous gene symbols") or "").split(";")]
            for s in syms:
                if s:
                    out.setdefault(s, []).append(gid)
    return out


def main() -> int:
    args = parse_args()

    gene_to_g2p = load_gene_to_g2p(args.g2p_csv)
    panel_symbols = set(gene_to_g2p)
    all_g2p_ids = sorted({g for ids in gene_to_g2p.values() for g in ids})
    print(f"G2P panel: {len(all_g2p_ids)} entries over {len(panel_symbols)} symbols", flush=True)

    df = pl.read_parquet(args.input_parquet)
    if "pmid" not in df.columns or "tiab" not in df.columns:
        raise SystemExit("--input_parquet must have 'pmid' and 'tiab' columns")
    pmids = set(df["pmid"].cast(pl.Utf8).to_list())
    print(f"input rows: {df.height}  unique pmids: {len(pmids)}", flush=True)

    gene_info = load_gene_info(args.gene_info)
    print(f"human genes in gene_info: {len(gene_info)}", flush=True)
    pub_genes = load_pubtator_genes(args.gene2pubtator, pmids, gene_info)
    print(f"pmids with >=1 PubTator human gene: {len(pub_genes)}", flush=True)

    matcher = None
    if args.hgnc:
        matcher = GeneNameMatcher.from_hgnc(args.hgnc, panel_symbols,
                                            family_stems=args.family_stems)
        print(f"HGNC names indexed: {len(matcher.name_to_symbols)} "
              f"families: {len(matcher.family_to_symbols)}", flush=True)

    cand_col: list[list[str]] = []
    src_col: list[list[str]] = []
    n_symbol = n_name = n_fallback = n_none = 0

    for pmid, tiab in zip(df["pmid"].cast(pl.Utf8).to_list(), df["tiab"].to_list()):
        by_symbol = pub_genes.get(pmid, set()) & panel_symbols
        by_name = (matcher.find(tiab or "") if matcher is not None else set()) - by_symbol
        # PubTator has no annotation for this PMID at all -> verbatim panel symbols in the text
        by_fallback: set[str] = set()
        if args.symbol_fallback and pmid not in pub_genes:
            by_fallback = find_symbols_verbatim(tiab or "", panel_symbols) - by_name
        if by_symbol:
            n_symbol += 1
        if by_name:
            n_name += 1
        if by_fallback:
            n_fallback += 1

        ids: dict[str, str] = {}
        for sym in by_symbol:
            for gid in gene_to_g2p.get(sym, []):
                ids[gid] = "symbol_match"
        for sym in by_name:
            for gid in gene_to_g2p.get(sym, []):
                ids.setdefault(gid, "name_match")
        for sym in by_fallback:
            for gid in gene_to_g2p.get(sym, []):
                ids.setdefault(gid, "symbol_fallback")

        if not ids:
            n_none += 1
            if args.keep_unmatched:
                ids = {gid: "fallback_full_panel" for gid in all_g2p_ids}
        ordered = sorted(ids)
        cand_col.append(ordered)
        src_col.append([ids[g] for g in ordered])

    df = df.with_columns([
        pl.Series("candidate_g2p_ids", cand_col, dtype=pl.List(pl.Utf8)),
        pl.Series("candidate_sources", src_col, dtype=pl.List(pl.Utf8)),
    ])

    kept = df.filter(pl.col("candidate_g2p_ids").list.len() > 0)
    n_cand = kept["candidate_g2p_ids"].list.len()

    print(f"\nrows with a symbol match : {n_symbol}")
    print(f"rows adding a name match : {n_name}")
    print(f"rows via symbol fallback : {n_fallback} (PubTator had no annotation for the PMID)")
    print(f"rows with no gene        : {n_none} "
          f"({'kept via full-panel fallback' if args.keep_unmatched else 'DROPPED'})")
    print(f"rows retained            : {kept.height} / {df.height} "
          f"({100 * kept.height / max(df.height, 1):.1f}%)")
    if kept.height:
        print(f"candidates per row       : mean {n_cand.mean():.2f}  median {n_cand.median():.0f} "
              f"max {n_cand.max()}")
        print(f"total (tiab, candidate) pairs to score: {int(n_cand.sum()):,} "
              f"(vs {df.height * len(all_g2p_ids):,} for the full panel)")

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    kept.write_parquet(args.out_parquet, compression="zstd")
    print(f"\nwrote {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
