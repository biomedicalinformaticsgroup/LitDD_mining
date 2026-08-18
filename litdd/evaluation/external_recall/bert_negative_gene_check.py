#!/usr/bin/env python3
"""Test whether the causative gene is mentioned in LitDD BERT-negative recall misses (R3.4).

LitDD's BERT (run over all of PubMed) classifies a paper positive only if it reads as a
gene–disease case mapping. A curated paper it scores negative is a miss — but if the paper
does not mention the G2P entry's gene at all, it carries no molecular confirmation (phenotype
described, causative gene unpublished — e.g. pre-discovery Marfan/Duchenne/Noonan reports),
which the pipeline deliberately excludes. Such papers can be removed from the recall
denominator on principled grounds.

For each BERT-negative miss this fetches the title+abstract (NCBI efetch) and tests whether
the gene **symbol or any previous/alias symbol** (DDG2P columns) appears (token boundary).
This is a rough abstract-level screen — full names and full text are not covered, so the
gene-absent count is an UPPER bound; PubTator over full text is the accurate follow-up (run
it on bert_negative_gene_absent_pmids.txt).

Outputs (gitignored revision/ area):
  bert_negative_gene_presence.csv      every miss: pmid, g2p, symbols, title, abstract, gene_in_tiab
  bert_negative_gene_present.csv       the gene-present subset (for manual review)
  bert_negative_gene_absent_pmids.txt  PMIDs with no gene mention in the abstract (-> PubTator)
"""
from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504])))
    return s


def efetch_tiab(sess, pmids, api_key) -> dict[str, tuple[str, str]]:
    """pmid -> (title, abstract) for a batch."""
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    for _ in range(3):
        try:
            r = sess.post(EFETCH, data=params, timeout=180)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            out = {}
            for art in root.findall(".//PubmedArticle"):
                pm = art.findtext(".//MedlineCitation/PMID")
                title = art.findtext(".//ArticleTitle") or ""
                abstract = " ".join((a.text or "") for a in art.findall(".//Abstract/AbstractText"))
                if pm:
                    out[pm] = (title, abstract)
            return out
        except (requests.RequestException, ET.ParseError):
            time.sleep(3)
    return {}


def gene_symbols(ddg2p_csv: str) -> dict[str, set[str]]:
    dd = pd.read_csv(ddg2p_csv, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    out: dict[str, set[str]] = {}
    for g, sym, prev in zip(dd["g2p id"], dd["gene symbol"], dd["previous gene symbols"]):
        syms = {sym.strip()} | {x.strip() for x in prev.replace(";", ",").split(",")}
        out[g] = {s for s in syms if s}
    return out


def mentions(symbols: set[str], text: str) -> bool:
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", text) for s in symbols)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--misses", default="revision/external_recall/deployed_misses.csv")
    ap.add_argument("--ddg2p", required=True)
    ap.add_argument("--out_dir", default="revision/external_recall")
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--batch_size", type=int, default=100)
    return ap.parse_args()


def main():
    args = parse_args()
    miss = pd.read_csv(args.misses, dtype=str)
    bn = miss[miss["category"] == "litdd_bert_negative"].copy()
    g2sym = gene_symbols(args.ddg2p)
    bn["symbols"] = bn["g2p"].map(lambda g: sorted(g2sym.get(g, set())))

    sess = session()
    pmids = sorted(set(bn["pmid"]))
    tiab: dict[str, tuple[str, str]] = {}
    sleep = 0.11 if args.api_key else 0.34
    for i in range(0, len(pmids), args.batch_size):
        tiab.update(efetch_tiab(sess, pmids[i:i + args.batch_size], args.api_key))
        time.sleep(sleep)
    print(f"fetched title+abstract for {len(tiab)}/{len(pmids)} PMIDs")

    bn["title"] = bn["pmid"].map(lambda p: tiab.get(p, ("", ""))[0])
    bn["abstract"] = bn["pmid"].map(lambda p: tiab.get(p, ("", ""))[1])
    bn["gene_in_tiab"] = [mentions(set(s), f"{t} {a}") for s, t, a in zip(bn["symbols"], bn["title"], bn["abstract"])]
    bn["symbols"] = bn["symbols"].map(lambda s: ";".join(s))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["pmid", "g2p", "symbols", "title", "abstract", "gene_in_tiab"]
    bn[cols].to_csv(out / "bert_negative_gene_presence.csv", index=False)
    bn[bn["gene_in_tiab"]][cols].to_csv(out / "bert_negative_gene_present.csv", index=False)
    absent = bn[~bn["gene_in_tiab"]]
    (out / "bert_negative_gene_absent_pmids.txt").write_text("\n".join(sorted(set(absent["pmid"]))))

    n = len(bn)
    print(f"BERT-negative miss pairs: {n}")
    print(f"  gene present in title+abstract: {int(bn['gene_in_tiab'].sum())} "
          f"({100 * bn['gene_in_tiab'].mean():.0f}%) -> bert_negative_gene_present.csv")
    print(f"  gene absent (no molecular confirmation in abstract): {int((~bn['gene_in_tiab']).sum())} "
          f"({100 * (~bn['gene_in_tiab']).mean():.0f}%) -> bert_negative_gene_absent_pmids.txt (run PubTator on full text)")


if __name__ == "__main__":
    main()
