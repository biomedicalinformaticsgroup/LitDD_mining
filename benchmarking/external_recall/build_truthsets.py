#!/usr/bin/env python3
"""Assemble external ground-truth (key -> PMID) sets for the LitDD recall evaluation
(Reviewer 3 R3.4 / Reviewer 2 C1/C2): the curated literature LitDD should recover.

Three sources, all dated before the August-2025 DDG2P that LitDD is built on:
  - premined : DDG2P `publications` column (PMIDs already curated for each G2P entry).
               Keyed by g2p_id (no disease matching needed).
  - hpoa     : phenotype.hpoa `reference` PMIDs, matched to DDG2P via OMIM
               (DDG2P `disease mim` <-> HPOA `OMIM:` database_id). HPOA has no MONDO, so
               OMIM is the bridge. Keyed by g2p_id (a disease may map to several entries).
  - clingen  : ClinGen gene -> evidence PMIDs (clingen_pmid_df.p). Keyed by gene symbol.

DDG2P MONDOs are completed from a newer DDG2P export **only** to backfill missing MONDOs
by g2p-id match (the newer file is otherwise substantially changed and is not used for
anything else).

Output: a tidy CSV `truthsets.csv` with columns: source, match_type, key, pmid.
(Default out dir is gitignored: it contains curated PMID lists.)
"""
from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import pandas as pd

PMID_RE = re.compile(r"\d+")
PMID_REF_RE = re.compile(r"PMID:\s*(\d+)")  # HPOA reference cells may list several refs


def _pmids(cell) -> list[str]:
    """Extract PMID integers from a string / list cell."""
    if isinstance(cell, (list, tuple)):
        return [str(int(x)) for x in cell if str(x).strip().isdigit()]
    return PMID_RE.findall(str(cell)) if cell is not None else []


def load_ddg2p(ddg2p_csv: str, mondo_backfill_csv: str | None) -> pd.DataFrame:
    dd = pd.read_csv(ddg2p_csv, dtype=str).fillna("")
    dd.columns = [c.strip() for c in dd.columns]
    if mondo_backfill_csv:
        bf = pd.read_csv(mondo_backfill_csv, dtype=str).fillna("")
        bf.columns = [c.strip() for c in bf.columns]
        m = dict(zip(bf["g2p id"], bf["disease MONDO"]))
        miss = dd["disease MONDO"].str.strip() == ""
        dd.loc[miss, "disease MONDO"] = dd.loc[miss, "g2p id"].map(m).fillna("")
        print(f"[ddg2p] MONDO backfilled for {int((miss & (dd['disease MONDO'] != '')).sum())} entries")
    return dd


def premined(dd: pd.DataFrame) -> pd.DataFrame:
    rows = [(gid, p) for gid, pub in zip(dd["g2p id"], dd["publications"]) for p in _pmids(pub)]
    out = pd.DataFrame(rows, columns=["key", "pmid"]).drop_duplicates()
    out.insert(0, "match_type", "g2p_id")
    out.insert(0, "source", "premined")
    return out


def hpoa(dd: pd.DataFrame, hpoa_path: str) -> pd.DataFrame:
    # OMIM -> list of g2p_ids
    omim_to_g2p: dict[str, list[str]] = {}
    for gid, mim in zip(dd["g2p id"], dd["disease mim"]):
        for m in _pmids(mim):  # disease mim is the OMIM number
            omim_to_g2p.setdefault(m, []).append(gid)

    rows = []
    cols = ["database_id", "disease_name", "qualifier", "hpo_id", "reference"]
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", usecols=range(5), names=cols, header=0, dtype=str)
    for db, ref in zip(df["database_id"].astype(str), df["reference"].astype(str)):
        if not db.startswith("OMIM:"):
            continue
        gids = omim_to_g2p.get(db.split(":", 1)[1].strip(), [])
        # a reference cell can list several refs ("PMID:a;PMID:b;OMIM:c") -> extract every PMID
        for pmid in PMID_REF_RE.findall(ref):
            for gid in gids:
                rows.append((gid, pmid))
    out = pd.DataFrame(rows, columns=["key", "pmid"]).drop_duplicates()
    out.insert(0, "match_type", "g2p_id")
    out.insert(0, "source", "hpoa")
    return out


def clingen(dd: pd.DataFrame, clingen_pickle: str, clingen_summary: str) -> pd.DataFrame:
    # Map ClinGen gene -> G2P ID via (gene, MONDO): a ClinGen gene-disease whose MONDO is in
    # DDG2P resolves to the DDG2P entry (g2p_id) with the same gene+MONDO. The gene's
    # evidence PMIDs are assigned to that g2p_id (per-disease, G2P-ID level).
    gm_to_g2p: dict[tuple[str, str], list[str]] = {}
    for gid, gene, mondo in zip(dd["g2p id"], dd["gene symbol"], dd["disease MONDO"]):
        if mondo.strip():
            gm_to_g2p.setdefault((gene, mondo.strip()), []).append(gid)

    summ = pd.read_csv(clingen_summary, dtype=str).fillna("")
    summ.columns = [c.strip() for c in summ.columns]
    gcol = next(c for c in summ.columns if c.upper().startswith("GENE SYMBOL"))
    mcol = next(c for c in summ.columns if "MONDO" in c.upper())
    gene_to_g2p: dict[str, set[str]] = {}
    for gene, mondo in zip(summ[gcol], summ[mcol]):
        for gid in gm_to_g2p.get((gene, mondo.strip()), []):
            gene_to_g2p.setdefault(gene, set()).add(gid)

    cp = pickle.load(open(clingen_pickle, "rb"))
    rows = [(gid, p) for gene, pmids in zip(cp["gene"], cp["pmids"])
            for gid in gene_to_g2p.get(gene, ()) for p in _pmids(pmids)]
    out = pd.DataFrame(rows, columns=["key", "pmid"]).drop_duplicates()
    out.insert(0, "match_type", "g2p_id")
    out.insert(0, "source", "clingen")
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ddg2p", required=True, help="August-2025 DDG2P CSV (publications, disease mim, disease MONDO)")
    ap.add_argument("--mondo_backfill", default=None, help="Newer DDG2P CSV — MONDO backfill by g2p-id only")
    ap.add_argument("--hpoa", required=True, help="phenotype.hpoa")
    ap.add_argument("--clingen_pickle", required=True, help="clingen_pmid_df.p (gene -> pmids)")
    ap.add_argument("--clingen_summary", required=True, help="Clingen-Gene-Disease-Summary CSV (gene, MONDO)")
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    dd = load_ddg2p(args.ddg2p, args.mondo_backfill)
    parts = [premined(dd), hpoa(dd, args.hpoa), clingen(dd, args.clingen_pickle, args.clingen_summary)]
    ts = pd.concat(parts, ignore_index=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts.to_csv(out / "truthsets.csv", index=False)

    print(f"\nWrote {len(ts):,} (key, pmid) ground-truth pairs -> {out}/truthsets.csv")
    for src, g in ts.groupby("source"):
        print(f"  {src:9s}: {len(g):>7,} pairs | {g['key'].nunique():>5,} keys | {g['pmid'].nunique():>7,} unique PMIDs")


if __name__ == "__main__":
    main()
