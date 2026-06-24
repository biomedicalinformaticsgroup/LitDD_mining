#!/usr/bin/env python3
"""Fetch gene annotations for a set of PMIDs directly from the PubTator3 API and write
them in ``gene2pubtator3`` format, so the gene-mention filter in ``final_data_clean.py``
can run on fresh per-abstract PubTator3 annotations instead of the bulk download.

Motivation (Reviewer 2 R2-C1 / Reviewer 3 R3.4): the gene-in-TIAB filter drops a large
fraction of score-passing mappings. Some of that attrition may be the bulk gene2pubtator3
file's coverage gaps rather than a true "gene absent" signal; querying PubTator3 per PMID
recovers genes the bulk file misses and lets us re-measure the filter's recall cost.

Adapted from a working PubTator3 annotation script: same endpoint and BioC-JSON parsing,
extended with batching, retry/back-off (robust to the transient NCBI errors that blocked
earlier attempts), an optional NCBI API key, and resume.

Output (one line per gene mention, tab-separated, matching gene2pubtator3 columns used by
final_data_clean.load_pubtator_genes):  PMID \t Gene \t <NCBI GeneID> \t <symbol>

    uv run python annotate_pubmed/pubtator_genes_api.py \
        --pmids pmids.txt --out pubtator_api_genes.tsv.gz
    # then:
    uv run python annotate_pubmed/final_data_clean.py --gene2pubtator pubtator_api_genes.tsv.gz ...
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PUBTATOR3_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"


def make_session() -> requests.Session:
    """Session with retry/back-off on the transient NCBI errors (429/5xx)."""
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def read_pmids(path: str) -> list[str]:
    if path.endswith(".csv"):
        col = "pmid"
        df = pd.read_csv(path)
        return [str(int(p)) for p in df[col if col in df.columns else df.columns[0]].dropna()]
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def fetch_genes(session, pmids: list[str], api_key: str | None, sleep: float):
    """Return {pmid: [(gene_id, symbol), ...]} for a batch of PMIDs from PubTator3."""
    params = {"pmids": ",".join(pmids)}
    if api_key:
        params["api_key"] = api_key
    try:
        resp = session.get(PUBTATOR3_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 — log and skip the batch; caller retries singly
        print(f"    PubTator3 error (n={len(pmids)}): {exc}", file=sys.stderr)
        return {}

    out: dict[str, list[tuple[str, str]]] = {}
    for pub in data.get("PubTator3", []):
        pmid = str(pub.get("pmid") or pub.get("id") or "").strip()
        if not pmid:
            continue
        genes = out.setdefault(pmid, [])
        seen = set()
        for passage in pub.get("passages", []):
            for ann in passage.get("annotations", []):
                if ann.get("infons", {}).get("type") != "Gene":
                    continue
                symbol = (ann.get("text") or "").strip()
                gene_id = str(ann.get("infons", {}).get("identifier") or "").strip().lstrip("@").replace("Gene_", "")
                if not symbol or not gene_id or gene_id == "-":
                    continue
                key = (gene_id, symbol)
                if key not in seen:
                    seen.add(key)
                    genes.append(key)
    time.sleep(sleep)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmids", required=True, help="Text file (one PMID per line) or CSV with a pmid column")
    ap.add_argument("--out", required=True, help="Output gene2pubtator3-format file (.tsv or .tsv.gz)")
    ap.add_argument("--batch_size", type=int, default=100, help="PMIDs per PubTator3 request")
    ap.add_argument("--api_key", default=os.getenv("NCBI_API_KEY"), help="NCBI API key (higher rate limit)")
    ap.add_argument("--sleep", type=float, default=None, help="Seconds between requests (default: 0.11 with key, 0.34 without)")
    return ap.parse_args()


def main():
    args = parse_args()
    sleep = args.sleep if args.sleep is not None else (0.11 if args.api_key else 0.34)
    session = make_session()

    pmids = read_pmids(args.pmids)
    # Resume: skip PMIDs already written.
    done: set[str] = set()
    opener = gzip.open if args.out.endswith(".gz") else open
    if os.path.exists(args.out):
        with opener(args.out, "rt") as f:
            done = {ln.split("\t", 1)[0] for ln in f if ln.strip()}
        print(f"Resuming: {len(done)} PMIDs already in {args.out}")
    todo = [p for p in pmids if p not in done]
    print(f"PMIDs: {len(pmids)} total, {len(todo)} to fetch, batch_size={args.batch_size}")

    n_pmids_with_genes = n_lines = 0
    with opener(args.out, "at") as out_f:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            res = fetch_genes(session, batch, args.api_key, sleep)
            for pmid, genes in res.items():
                if genes:
                    n_pmids_with_genes += 1
                for gene_id, symbol in genes:
                    out_f.write(f"{pmid}\tGene\t{gene_id}\t{symbol}\n")
                    n_lines += 1
            out_f.flush()
            if (i // args.batch_size) % 20 == 0:
                print(f"  {min(i + len(batch), len(todo))}/{len(todo)} PMIDs, "
                      f"{n_lines} gene mentions so far", flush=True)

    print(f"Done: {n_lines} gene mentions for {n_pmids_with_genes} PMIDs -> {args.out}")


if __name__ == "__main__":
    main()
