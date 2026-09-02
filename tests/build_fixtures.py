#!/usr/bin/env python3
"""Build small fixtures for tests/test_final_data_clean.py.

Reads from the (large) reference clean_pipeline data to slice tiny self-contained
fixtures into tests/fixtures/. Run once whenever the source data is regenerated.
Point it at the reference data via the LITDD_REF_DIR environment variable, e.g.
  LITDD_REF_DIR=/path/to/clean_pipeline python tests/build_fixtures.py

Outputs (all under tests/fixtures/):
  llm_shard_sample.parquet
  g2p_sample.csv
  gene2pubtator_sample.tsv.gz
  gene_info_sample.gz
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REF = Path(os.environ.get("LITDD_REF_DIR", "reference_data/clean_pipeline"))
LLM = REF / "litdd/pipeline/data/crossencoded_shards_llm/pubmed_bert_positive_crossencoded_shard0-of-4__llm.parquet"
G2P = REF / "litdd/training/G2P_DD_2025-02-15.csv"
GENE2PUBTATOR = REF / "litdd/pipeline/data/gene2pubtator3"
GENE_INFO = REF / "litdd/pipeline/data/GNorm2/gene_info"

OUT = Path(__file__).resolve().parent / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[1/4] reading first batch of {LLM.name}…")
    pf = pq.ParquetFile(str(LLM))
    batch = next(pf.iter_batches(batch_size=2000, columns=["pmid", "llm_dis_map", "top5_cross"]))
    df = batch.to_pandas()
    df = df[df["llm_dis_map"].notna()]
    df = df[df["llm_dis_map"].astype(str).str.startswith("G2P")]
    df = df.head(40).reset_index(drop=True)
    print(f"        kept {len(df)} rows")

    out_parquet = OUT / "llm_shard_sample.parquet"
    df.to_parquet(out_parquet, compression="zstd")
    print(f"        wrote {out_parquet} ({out_parquet.stat().st_size/1024:.1f} KB)")

    pmids = {str(p) for p in df["pmid"].tolist()}
    g2p_ids = set()
    for v in df["llm_dis_map"]:
        for tok in str(v).split(";"):
            tok = tok.strip().strip("'\"")
            if tok.startswith("G2P"):
                g2p_ids.add(tok)
    for top5 in df["top5_cross"]:
        if top5 is None:
            continue
        for item in top5:
            label = item.get("label") if isinstance(item, dict) else ""
            if label and label.startswith("G2P"):
                g2p_ids.add(label.split(" ")[0].strip())
    print(f"        PMIDs={len(pmids)} G2P_IDs={len(g2p_ids)}")

    print("[2/4] slicing G2P CSV…")
    g2p_full = pd.read_csv(G2P, dtype=str, keep_default_na=False)
    g2p_slice = g2p_full[g2p_full["g2p id"].isin(g2p_ids)].copy()
    # Add 5 extra G2P rows to test hallucination filtering
    extras = g2p_full[~g2p_full["g2p id"].isin(g2p_ids)].head(5)
    g2p_out = pd.concat([g2p_slice, extras], ignore_index=True)
    g2p_csv = OUT / "g2p_sample.csv"
    g2p_out.to_csv(g2p_csv, index=False)
    print(f"        wrote {g2p_csv} ({g2p_csv.stat().st_size/1024:.1f} KB, {len(g2p_out)} rows)")

    print("[3/4] slicing gene2pubtator (this scans 2.9GB once)…")
    g2p_pubtator_rows = []
    referenced_geneids = set()
    with open(GENE2PUBTATOR, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4 or parts[0] not in pmids:
                continue
            g2p_pubtator_rows.append(line.rstrip("\n"))
            eid = parts[2].split(";")[0].strip() if parts[2] else ""
            if eid.isdigit():
                referenced_geneids.add(eid)
    g2p_tsv = OUT / "gene2pubtator_sample.tsv.gz"
    with gzip.open(g2p_tsv, "wt", encoding="utf-8") as f:
        f.write("\n".join(g2p_pubtator_rows) + ("\n" if g2p_pubtator_rows else ""))
    print(f"        wrote {g2p_tsv} ({g2p_tsv.stat().st_size/1024:.1f} KB, {len(g2p_pubtator_rows)} rows)")

    print("[4/4] slicing gene_info (taxid=9606)…")
    gi_lines = []
    with open(GENE_INFO, "r", encoding="utf-8") as f:
        header = f.readline()
        gi_lines.append(header.rstrip("\n"))
        for line in f:
            parts = line.split("\t", 3)
            if len(parts) < 3 or parts[0] != "9606":
                continue
            if parts[1] in referenced_geneids:
                gi_lines.append(line.rstrip("\n"))
    gi_gz = OUT / "gene_info_sample.gz"
    with gzip.open(gi_gz, "wt", encoding="utf-8") as f:
        f.write("\n".join(gi_lines) + "\n")
    print(f"        wrote {gi_gz} ({gi_gz.stat().st_size/1024:.1f} KB, {len(gi_lines)-1} symbols)")

    print("\n[done] fixtures ready under tests/fixtures/")


if __name__ == "__main__":
    main()
