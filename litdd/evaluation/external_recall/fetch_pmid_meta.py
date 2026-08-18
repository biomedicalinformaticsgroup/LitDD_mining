#!/usr/bin/env python3
"""Fetch publication year + publication types + title for a set of PMIDs via NCBI esummary.

Used to (a) exclude pre-1980 papers from the recall denominator (LitDD's BERT step filters
pubdate > 1980, so older papers can never be retrieved) and (b) characterise misses by
publication type (e.g. reviews / functional studies that LitDD's case-report screen omits).

Output CSV: pmid, year, pubtypes, title.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504])))
    return s


def fetch(sess, pmids, api_key):
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    r = sess.post(ESUMMARY, data=params, timeout=60)  # POST avoids URI-too-long
    r.raise_for_status()
    res = r.json().get("result", {})
    out = []
    for pid in res.get("uids", []):
        d = res[pid]
        year = (d.get("pubdate", "") or "")[:4]
        pubtypes = "|".join(d.get("pubtype", []) or [])
        lang = "|".join(d.get("lang", []) or [])
        # GeneReviews/StatPearls are NCBI Bookshelf chapters: empty `source`, name in `booktitle`
        source = d.get("booktitle", "") or d.get("source", "") or d.get("fulljournalname", "")
        out.append((pid, year, pubtypes, lang, source, d.get("title", "")))
    return out


def read_pmids(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path, dtype=str)
        col = "pmid" if "pmid" in df.columns else df.columns[0]
        return sorted(set(df[col].dropna()))
    return sorted({ln.strip() for ln in open(path) if ln.strip()})


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pmids", required=True, help="CSV with a pmid column, or a .txt of PMIDs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--api_key", default=os.getenv("NCBI_API_KEY"))
    return ap.parse_args()


def main():
    args = parse_args()
    pmids = read_pmids(args.pmids)
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, dtype=str)["pmid"])
    todo = [p for p in pmids if p not in done]
    print(f"{len(pmids)} PMIDs, {len(todo)} to fetch")
    sess = session()
    sleep = 0.11 if args.api_key else 0.34
    cols = ["pmid", "year", "pubtypes", "lang", "source", "title"]

    def checkpoint(rows):
        df = pd.DataFrame(rows, columns=cols)
        if os.path.exists(args.out):
            df = pd.concat([pd.read_csv(args.out, dtype=str), df], ignore_index=True).drop_duplicates("pmid")
        df.to_csv(args.out, index=False)

    rows = []
    for i in range(0, len(todo), args.batch_size):
        for attempt in range(4):  # tolerate transient 5xx without losing progress
            try:
                rows += fetch(sess, todo[i:i + args.batch_size], args.api_key)
                break
            except Exception as e:  # noqa: BLE001 - network flakiness; checkpoint and retry
                if attempt == 3:
                    print(f"  batch {i} failed ({e}); checkpointing, rerun to resume")
                    checkpoint(rows)
                    raise
                time.sleep(5)
        time.sleep(sleep)
        if i % (args.batch_size * 10) == 0:
            checkpoint(rows)
            rows = []
            print(f"  {min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    checkpoint(rows)
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
