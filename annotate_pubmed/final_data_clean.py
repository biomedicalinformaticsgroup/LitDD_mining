#!/usr/bin/env python3
"""Filter LLM mapping output and emit the final (PMID, G2P_ID) table.

For each row in the LLM output parquet, keep a (PMID, G2P_ID) pair only when
*all* of the following hold:

  1. The LLM produced a non-``NO MATCH`` answer that starts with ``G2P``.
  2. The G2P_ID exists in the G2P CSV (no LLM hallucination).
  3. The cross-encoder ``top5_cross`` score for this G2P_ID is ≥ ``--score_cutoff``.
  4. At least one gene linked to that G2P_ID is mentioned in the abstract,
     as recorded by GNorm2 in PubTator's ``gene2pubtator3`` file. To stay
     robust to local synonym variation, gene2pubtator's NCBI Gene IDs are
     resolved to canonical symbols via ``gene_info.gz`` (``tax_id == 9606``).

This is a re-implementation of the original notebook-style cleaner: it streams
the LLM parquet via ``pyarrow.parquet.iter_batches`` (constant memory),
takes paths via CLI, and emits CSV. Gene-symbol normalisation uses NCBI
``gene_info`` rather than the raw GNorm2 mention text, matching the
behaviour the published results were generated under.

Usage:
    python final_data_clean.py \\
        --llm_file pubmed_..._llm.parquet \\
        --g2p_file G2P_DD_2025-02-15.csv \\
        --gene2pubtator gene2pubtator3.gz \\
        --gene_info gene_info.gz \\
        --output_csv final_cleaned_data.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import sys
from typing import Dict, List, Set

import pyarrow.parquet as pq

GENE_INFO_TAXID_HUMAN = "9606"
G2P_ID_RE = re.compile(r"^(G2P\d+)\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llm_file", required=True, help="LLM-output parquet file.")
    p.add_argument("--g2p_file", required=True, help="G2P DD CSV.")
    p.add_argument("--gene2pubtator", required=True, help="gene2pubtator3 .gz file (PubTator).")
    p.add_argument(
        "--gene_info",
        required=False,
        default=None,
        help="NCBI gene_info.gz; if provided, gene2pubtator NCBI Gene IDs are mapped "
             "to canonical symbols (tax_id=9606). Strongly recommended.",
    )
    p.add_argument("--score_cutoff", type=float, default=0.9,
                   help="Minimum top5_cross score to keep a row (default 0.9).")
    p.add_argument("--output_csv", required=True, help="Output CSV (PMID, G2P_IDs).")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def load_g2p_maps(path: str) -> Dict[str, List[str]]:
    """g2p id -> [gene symbol, *previous gene symbols]."""
    mp: Dict[str, List[str]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g2p_id = (row.get("g2p id") or "").strip()
            if not g2p_id:
                continue
            gene = (row.get("gene symbol") or "").strip()
            prev = (row.get("previous gene symbols") or "").strip()
            symbols = [gene] if gene else []
            symbols.extend(p.strip() for p in prev.split(";") if p.strip())
            if symbols:
                mp[g2p_id] = symbols
    return mp


def load_gene_info(path: str) -> Dict[str, str]:
    """NCBI GeneID (str) -> canonical Symbol, restricted to human (tax_id=9606)."""
    mp: Dict[str, str] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        try:
            i_tax = header.index("tax_id")
            i_gid = header.index("GeneID")
            i_sym = header.index("Symbol")
        except ValueError as e:
            raise SystemExit(f"[ERROR] gene_info missing expected column: {e}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_tax, i_gid, i_sym):
                continue
            if parts[i_tax] != GENE_INFO_TAXID_HUMAN:
                continue
            mp[parts[i_gid]] = parts[i_sym]
    return mp


def load_llm_parquet(path: str):
    """Stream pmid -> llm_dis_map and pmid -> {g2p_id: top5_score}."""
    pmid_to_llm: Dict[str, str] = {}
    pmid_to_top5: Dict[str, Dict[str, float]] = {}
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(columns=["pmid", "llm_dis_map", "top5_cross"]):
        pmids = batch.column(0).to_pylist()
        llms = batch.column(1).to_pylist()
        top5s = batch.column(2).to_pylist()
        for pmid, llm_val, top5_val in zip(pmids, llms, top5s):
            pmid_str = str(pmid)
            pmid_to_llm[pmid_str] = llm_val
            scores: Dict[str, float] = pmid_to_top5.setdefault(pmid_str, {})
            for item in top5_val or []:
                label = item.get("label") if isinstance(item, dict) else None
                score = item.get("score") if isinstance(item, dict) else None
                m = G2P_ID_RE.search(label or "")
                if m:
                    scores[m.group(1)] = score
    return pmid_to_llm, pmid_to_top5


def load_pubtator_genes(
    path: str, pmids: Set[str], gene_info: Dict[str, str] | None
) -> Dict[str, Set[str]]:
    """pmid (str) -> set of gene symbols mentioned in the abstract.

    If ``gene_info`` is provided, GNorm2 NCBI Gene IDs (column 2) are mapped to
    canonical symbols. Otherwise we fall back to the raw mention text in
    column 3 — works but matches less reliably against G2P symbols.
    """
    out: Dict[str, Set[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            pmid = parts[0]
            if pmid not in pmids:
                continue
            symbols: List[str] = []
            if gene_info is not None:
                # entity_id may be "12345" or "12345;67890"; first wins
                eid = parts[2].split(";")[0].strip() if parts[2] else ""
                sym = gene_info.get(eid)
                if sym:
                    symbols.append(sym)
            else:
                # Fallback: use the mention text column
                symbols = [v.strip() for v in (parts[3] or "").split("|") if v.strip()]
            if symbols:
                out.setdefault(pmid, set()).update(symbols)
    return out


def main() -> int:
    args = parse_args()

    for label, p in [
        ("LLM parquet", args.llm_file),
        ("G2P CSV", args.g2p_file),
        ("gene2pubtator", args.gene2pubtator),
    ]:
        if not os.path.exists(p):
            print(f"[ERROR] {label} not found: {p}", file=sys.stderr)
            return 1
    if args.gene_info and not os.path.exists(args.gene_info):
        print(f"[ERROR] gene_info not found: {args.gene_info}", file=sys.stderr)
        return 1

    g2p_genes = load_g2p_maps(args.g2p_file)
    valid_g2p_ids = set(g2p_genes.keys())

    gene_info = load_gene_info(args.gene_info) if args.gene_info else None
    if gene_info is not None:
        print(f"[INFO] gene_info: {len(gene_info)} human GeneID→Symbol entries")

    pmid_to_llm, pmid_to_top5 = load_llm_parquet(args.llm_file)
    pubtator_genes = load_pubtator_genes(args.gene2pubtator, set(pmid_to_llm), gene_info)

    total = kept = 0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["PMID", "G2P_IDs"])
        for pmid, llm_val in pmid_to_llm.items():
            if not llm_val or llm_val == "NO MATCH":
                continue
            mentioned = pubtator_genes.get(pmid, set())
            scores = pmid_to_top5.get(pmid, {})
            if args.debug:
                print(f"\nPMID={pmid} mentioned={sorted(mentioned)} llm={llm_val}")
            for raw in str(llm_val).split(";"):
                g2p = raw.strip().strip("'\"")
                if not g2p:
                    continue
                total += 1
                # 1. valid G2P ID (no LLM hallucination)
                if g2p not in valid_g2p_ids:
                    if args.debug:
                        print(f"  {g2p}\tHALLUCINATED")
                    continue
                # 2. score cutoff (always applied if cutoff > 0)
                score = scores.get(g2p)
                if args.score_cutoff > 0 and (score is None or score < args.score_cutoff):
                    if args.debug:
                        s = "None" if score is None else f"{score:.2f}"
                        print(f"  {g2p}\tSCORE_BELOW_CUTOFF (score={s} < {args.score_cutoff})")
                    continue
                # 3. gene match (always applied — pulled out of score_cutoff conditional)
                if not any(g in mentioned for g in g2p_genes[g2p]):
                    if args.debug:
                        print(f"  {g2p}\tGENE_NOT_MENTIONED (g2p_genes={g2p_genes[g2p]})")
                    continue
                kept += 1
                writer.writerow([pmid, g2p])
                if args.debug:
                    print(f"  {g2p}\tVALID")

    print(f"[INFO] Loaded:           {args.llm_file}")
    print(f"[INFO] Total mappings:   {total}")
    print(f"[INFO] Valid mappings:   {kept}")
    print(f"[INFO] Wrote:            {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
