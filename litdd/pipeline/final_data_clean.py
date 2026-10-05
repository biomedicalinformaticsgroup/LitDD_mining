#!/usr/bin/env python3
"""Filter the adjudication output down to the final (PMID, G2P id) table.

Reads
    ``--llm_file``            adjudication parquet from ``llm_map.py`` (columns ``pmid``,
                              ``llm_dis_map``, and ``candidates`` when present)
    ``--candidates_parquet``  ``candidates.parquet`` from ``gene_candidates.py`` (columns
                              ``pmid``, ``candidate_g2p_ids``)
    ``--g2p_file``            the G2P panel CSV that defines the valid ids
    ``--gene2pubtator``       gene2pubtator3 bulk file (optional, see below)
    ``--gene_info``           NCBI gene_info (optional, see below)

Writes
    ``--output_csv``    ``PMID,G2P_IDs`` with one row per accepted (PMID, G2P id) pair
    ``--no_match_csv``  ``pmid,genes_mentioned,n_candidates`` for abstracts whose answer was
                        empty or ``NO MATCH`` (optional)

An answer may name several ids separated by ``;``. Each id is accepted when it is a
``g2p id`` of the panel CSV and it is one of the candidates the gene gate offered for that
abstract. Ids failing either test are dropped and counted in the summary.

``genes_mentioned`` in the no-match file lists the human gene symbols PubTator annotated
for the abstract, resolved through ``gene_info``; it is empty when either file is not given.
``n_candidates`` is the number of candidates the adjudicator saw, taken from the
``candidates`` column of the LLM parquet or, when that column is absent, from the
candidates parquet.

The LLM parquet is read in row-group batches, so memory does not grow with its size.

Usage:
    python -m litdd.pipeline.final_data_clean \\
        --llm_file llm_all.parquet \\
        --g2p_file G2P_DD.csv \\
        --candidates_parquet candidates.parquet \\
        --gene2pubtator gene2pubtator3.gz \\
        --gene_info gene_info.gz \\
        --output_csv final.csv \\
        --no_match_csv nomatch.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import pyarrow.parquet as pq

from litdd import genes

logger = logging.getLogger("litdd.final_data_clean")

NO_MATCH = "NO MATCH"
LLM_COLUMNS = ("pmid", "llm_dis_map")
NO_MATCH_FIELDS = ("pmid", "genes_mentioned", "n_candidates")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface of the cleaning stage."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llm_file", required=True, help="Adjudication parquet from llm_map.py.")
    p.add_argument("--g2p_file", required=True, help="G2P panel CSV defining the valid ids.")
    p.add_argument("--candidates_parquet", required=True,
                   help="candidates.parquet from gene_candidates.py (pmid, candidate_g2p_ids).")
    p.add_argument("--gene2pubtator", default=None,
                   help="gene2pubtator3 bulk file; fills genes_mentioned in --no_match_csv "
                        "(requires --gene_info).")
    p.add_argument("--gene_info", default=None,
                   help="NCBI gene_info file mapping GeneID to symbol (human rows only are used).")
    p.add_argument("--output_csv", required=True, help="Output CSV with columns PMID, G2P_IDs.")
    p.add_argument("--no_match_csv", default=None,
                   help="Write abstracts whose answer was empty or NO MATCH to this CSV.")
    p.add_argument("--debug", action="store_true", help="Log the decision for every mapping.")
    return p.parse_args(argv)


def load_candidate_ids(path: str) -> dict[str, set[str]]:
    """Return ``{pmid: set of G2P ids}`` offered by the gene gate, read from candidates.parquet."""
    out: dict[str, set[str]] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["pmid", "candidate_g2p_ids"]):
        for pmid, ids in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            out.setdefault(str(pmid), set()).update(str(i) for i in (ids or []))
    return out


def split_answer(answer: str) -> list[str]:
    """Split a ``;``-separated answer into ids, stripping whitespace and quote characters."""
    ids = []
    for raw in str(answer).split(";"):
        g2p = raw.strip().strip("'\"")
        if g2p:
            ids.append(g2p)
    return ids


def iter_llm_rows(path: str):
    """Yield ``(pmid, answer, n_candidates)`` per row of the LLM parquet, one batch at a time.

    ``n_candidates`` is the length of the ``candidates`` list, or ``None`` when the file has
    no such column.
    """
    pf = pq.ParquetFile(path)
    has_candidates = "candidates" in pf.schema_arrow.names
    columns = list(LLM_COLUMNS) + (["candidates"] if has_candidates else [])
    for batch in pf.iter_batches(columns=columns):
        pmids = batch.column(0).to_pylist()
        answers = batch.column(1).to_pylist()
        counts = ([len(c or []) for c in batch.column(2).to_pylist()] if has_candidates
                  else [None] * len(pmids))
        yield from zip((str(p) for p in pmids), answers, counts)


def genes_mentioned(gene2pubtator: str | None, gene_info: str | None,
                    pmids: set[str]) -> dict[str, set[str]]:
    """Return ``{pmid: gene symbols}`` for ``pmids`` when both resource paths are given."""
    if not (gene2pubtator and gene_info and pmids):
        return {}
    info = genes.load_gene_info(gene_info)
    logger.info("gene_info: %d human GeneID to symbol entries", len(info))
    return genes.load_pubtator_genes(gene2pubtator, pmids, info)


def write_no_match(path: str, rows: list[dict[str, object]],
                   mentioned: dict[str, set[str]]) -> None:
    """Write the no-match CSV, filling ``genes_mentioned`` from ``mentioned``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(NO_MATCH_FIELDS))
        w.writeheader()
        for row in rows:
            row["genes_mentioned"] = ";".join(sorted(mentioned.get(str(row["pmid"]), set())))
            w.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(message)s", stream=sys.stderr)

    # Every input path must exist before any work starts.
    paths = [("LLM parquet", args.llm_file), ("G2P CSV", args.g2p_file),
             ("candidates parquet", args.candidates_parquet),
             ("gene2pubtator", args.gene2pubtator), ("gene_info", args.gene_info)]
    for label, path in paths:
        if path and not os.path.exists(path):
            logger.error("%s not found: %s", label, path)
            return 1
    if bool(args.gene2pubtator) != bool(args.gene_info):
        logger.error("--gene2pubtator and --gene_info must be given together")
        return 1

    valid_ids = genes.g2p_ids(args.g2p_file)
    candidate_ids = load_candidate_ids(args.candidates_parquet)
    logger.info("candidate sets: %d abstracts", len(candidate_ids))

    total = kept = dropped_invalid = dropped_not_offered = 0
    no_match_rows: list[dict[str, object]] = []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["PMID", "G2P_IDs"])
        for pmid, answer, n_candidates in iter_llm_rows(args.llm_file):
            offered = candidate_ids.get(pmid, set())
            # Empty and NO MATCH answers go to the no-match file and contribute no mapping.
            if not answer or answer == NO_MATCH:
                if args.no_match_csv:
                    no_match_rows.append({
                        "pmid": pmid,
                        "genes_mentioned": "",
                        "n_candidates": len(offered) if n_candidates is None else n_candidates,
                    })
                continue
            logger.debug("PMID=%s offered=%s llm=%s", pmid, sorted(offered), answer)
            for g2p in split_answer(answer):
                total += 1
                # The id must exist in the panel and must have been offered for this abstract.
                if g2p not in valid_ids:
                    dropped_invalid += 1
                    logger.debug("  %s\tNOT_IN_PANEL", g2p)
                    continue
                if g2p not in offered:
                    dropped_not_offered += 1
                    logger.debug("  %s\tNOT_OFFERED_AS_CANDIDATE", g2p)
                    continue
                kept += 1
                writer.writerow([pmid, g2p])
                logger.debug("  %s\tVALID", g2p)

    logger.info("Loaded:                 %s", args.llm_file)
    logger.info("Total mappings:         %d", total)
    logger.info("Dropped, not in panel:  %d", dropped_invalid)
    logger.info("Dropped, not offered:   %d", dropped_not_offered)
    logger.info("Valid mappings:         %d", kept)
    logger.info("Wrote:                  %s", args.output_csv)

    if args.no_match_csv and no_match_rows:
        mentioned = genes_mentioned(args.gene2pubtator, args.gene_info,
                                    {str(r["pmid"]) for r in no_match_rows})
        write_no_match(args.no_match_csv, no_match_rows, mentioned)
        logger.info("NO MATCH abstracts:     %d -> %s", len(no_match_rows), args.no_match_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
