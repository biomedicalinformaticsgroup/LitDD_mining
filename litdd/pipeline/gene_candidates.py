"""Gene gate: restrict each screen-positive abstract to the G2P entries whose gene it names.

Reads the screened parquet (columns ``pmid``, ``tiab``), the G2P export CSV, the
``gene2pubtator3`` bulk file, ``hgnc_complete_set.txt`` and a MONDO OBO release. Every gene is
resolved by HGNC identifier (``litdd.gene_resolution``): PubTator3 GeneIDs through HGNC
``entrez_id``, HGNC descriptive names, and, where PubTator3 verified no panel gene in the text,
panel symbols written verbatim. Disease names and disease abbreviations drawn from MONDO are not
gene evidence.

Writes the input rows that have at least one candidate, with two list columns added:
``candidate_g2p_ids`` (sorted G2P ids) and ``candidate_sources`` (``symbol_match``,
``name_match`` or ``symbol_fallback`` per id). With ``--audit_prefix`` it also writes
``<prefix>_symbols.tsv`` (every symbol considered for the verbatim dictionary and why it was
kept or excluded), ``<prefix>_names.tsv`` (descriptive names removed as disease names) and
``<prefix>_stats.json`` (per-rule counts over the corpus).

Example::

    python -m litdd.pipeline.gene_candidates \\
        --input_parquet data/pubmed_bert_positive.parquet \\
        --g2p_csv data/G2P_DD.csv \\
        --gene2pubtator data/gene2pubtator3.gz \\
        --hgnc data/reference/hgnc_complete_set.txt \\
        --mondo_obo data/reference/mondo.obo \\
        --audit_prefix data/gate_audit/gate \\
        --out_parquet data/candidates.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import polars as pl

from litdd.gene_resolution import DiseaseLexicon, HgncPanel, resolve_candidates
from litdd.genes import load_pubtator_gene_ids

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_parquet", required=True, help="screened parquet with pmid and tiab columns")
    p.add_argument("--g2p_csv", required=True, help="G2P export CSV with 'g2p id' and 'hgnc id' columns")
    p.add_argument("--gene2pubtator", required=True, help="gene2pubtator3 bulk file (.gz or TSV)")
    p.add_argument("--hgnc", required=True, help="hgnc_complete_set.txt")
    p.add_argument("--mondo_obo", required=True, help="MONDO OBO release, the source of disease names")
    p.add_argument("--audit_prefix", default=None,
                   help="write <prefix>_symbols.tsv, <prefix>_names.tsv and <prefix>_stats.json")
    p.add_argument("--out_parquet", required=True)
    return p.parse_args(argv)


def write_audit(prefix: str, panel: HgncPanel, report: dict) -> None:
    """Write the symbol and name audit tables and the stats report under ``prefix``."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    pl.DataFrame(panel.symbol_audit).write_csv(f"{prefix}_symbols.tsv", separator="\t")
    if panel.name_audit:
        pl.DataFrame(panel.name_audit).write_csv(f"{prefix}_names.tsv", separator="\t")
    with open(f"{prefix}_stats.json", "w") as f:
        json.dump(report, f, indent=1)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = parse_args(argv)

    df = pl.read_parquet(args.input_parquet)
    if "pmid" not in df.columns or "tiab" not in df.columns:
        raise SystemExit("--input_parquet must have 'pmid' and 'tiab' columns")
    logger.info("input rows: %d", df.height)

    # Disease lexicons: every MONDO term for aliases, names and mentions; the germline lineage
    # for the abbreviation context rule.
    lexicon = DiseaseLexicon.from_obo(args.mondo_obo)
    context_lexicon = lexicon.germline_lineage()
    panel = HgncPanel.from_files(args.hgnc, args.g2p_csv, lexicon)
    logger.info("MONDO disease terms: %d; panel genes by HGNC ID: %d (%d entries; %d G2P rows "
                "with an HGNC id not in HGNC)", len(lexicon.labels), len(panel.entries),
                sum(map(len, panel.entries.values())), len(panel.unresolved_g2p))

    pmids = df["pmid"].cast(pl.Utf8).to_list()
    pubtator = load_pubtator_gene_ids(args.gene2pubtator, set(pmids))
    logger.info("pmids with at least one PubTator3 gene annotation: %d", len(pubtator))

    cand_col, src_col, report = resolve_candidates(
        zip(pmids, df["tiab"].to_list()), pubtator, panel, lexicon, context_lexicon)
    for row in report["symbol_audit"]:
        logger.info("  %-22s %-60s %6d", row["source"], row["outcome"], row["n"])
    logger.info("  descriptive names removed as disease names: %d", report["names_removed_as_disease"])
    if args.audit_prefix:
        write_audit(args.audit_prefix, panel, report)

    df = df.with_columns([
        pl.Series("candidate_g2p_ids", cand_col, dtype=pl.List(pl.Utf8)),
        pl.Series("candidate_sources", src_col, dtype=pl.List(pl.Utf8)),
    ])
    kept = df.filter(pl.col("candidate_g2p_ids").list.len() > 0)
    logger.info("rows retained: %d / %d", kept.height, df.height)
    if kept.height:
        logger.info("(tiab, candidate) pairs: %d", int(kept["candidate_g2p_ids"].list.len().sum()))
    logger.info("disease-abbreviation context declined %d (abstract, symbol) pairs; "
                "disease-name mentions dropped %d",
                report["disease_context_abstract_symbol_pairs"], report["disease_mentions_dropped"])

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    kept.write_parquet(args.out_parquet, compression="zstd")
    logger.info("wrote %s", args.out_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
