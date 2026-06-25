#!/usr/bin/env python3
"""Assemble external ground-truth (g2p_id -> PMID) sets for the LitDD recall evaluation
(Reviewer 3 R3.4 / Reviewer 2 C1/C2): the curated literature LitDD should recover.

Three sources, all dated before the August-2025 DDG2P that LitDD is built on, and all
restricted to disorders in DDG2P (matched at the disease/G2P-ID level):
  - premined : DDG2P `publications` column (PMIDs already curated for each G2P entry).
  - hpoa     : phenotype.hpoa `reference` PMIDs, matched to DDG2P via OMIM (DDG2P
               `disease mim` <-> HPOA `OMIM:` database_id); multi-PMID reference cells
               are fully parsed.
  - clingen  : ClinGen **case-level (genetic) evidence** PMIDs only — parsed from the
               `genetic_evidence_*` table exports per gene-disease, matched to DDG2P by
               MONDO (from the HGNC_*_MONDO_* export dir). The `experimental_evidence`
               (functional / model-organism) tables are deliberately excluded, matching
               the manuscript's "case level evidence" definition.

The recall universe (all sources incl. G2P) is restricted to **leaf MONDOs** — disease
terms with no MONDO disease subclass. This keeps single gene-diseases (e.g. SMC1A Cornelia
de Lange) and drops broad grouping terms (e.g. "dilated cardiomyopathy", "epileptic
encephalopathy", "skeletal dysplasia") whose curated literature spans many genes and would
otherwise be assigned to one arbitrary G2P entry. (MONDO's explicit gene-grounding axiom
`has material basis in germline mutation in <GENE>` is the ideal signal but is absent for
most leaf diseases that should carry it, so the leaf-vs-grouping structure is used instead.)

DDG2P MONDOs are completed from a newer DDG2P export **only** to backfill missing MONDOs
by g2p-id match (the newer file is otherwise unchanged-from used).

`--exclude_pmids` removes PMIDs present in the train/test annotation set, so recall is
measured only on held-out literature (standard generalisation evaluation).

Output: a tidy CSV `truthsets.csv` with columns: source, match_type, key, pmid.
(Default out dir is gitignored: it contains curated PMID lists.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PMID_RE = re.compile(r"\d+")
PMID_REF_RE = re.compile(r"PMID:\s*(\d+)")  # HPOA references & ClinGen evidence cells


def _pmids(cell) -> list[str]:
    if isinstance(cell, (list, tuple)):
        return [str(int(x)) for x in cell if str(x).strip().isdigit()]
    return PMID_RE.findall(str(cell)) if cell is not None else []


def leaf_mondos(mondo_json: str) -> set[str]:
    """MONDO terms with no MONDO disease subclass (leaves) — i.e. single gene-diseases,
    not broad grouping terms. A grouping term (e.g. 'skeletal dysplasia') has children."""
    g = json.load(open(mondo_json))["graphs"][0]
    parents = {
        "MONDO:" + e["obj"].rsplit("_", 1)[-1]
        for e in g["edges"]
        if e.get("pred") in ("is_a", "rdfs:subClassOf")
        and "MONDO_" in e.get("sub", "") and "MONDO_" in e.get("obj", "")
    }
    all_m = {"MONDO:" + n["id"].rsplit("_", 1)[-1] for n in g["nodes"] if "/MONDO_" in n["id"]}
    return all_m - parents


def restrict_to_leaf(dd: pd.DataFrame, leaves: set[str]):
    """Keep only DDG2P entries whose disease MONDO is a leaf; return (kept, excluded)."""
    mo = dd["disease MONDO"].str.strip()
    keep = mo.isin(leaves)
    excl = dd.loc[~keep].copy()
    excl["reason"] = mo.loc[~keep].map(lambda m: "no_mondo" if not m else "grouping_mondo")
    cols = ["g2p id", "gene symbol", "disease name", "disease MONDO", "reason"]
    return dd.loc[keep].copy(), excl[cols]


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
    return _frame(rows, "premined")


def hpoa(dd: pd.DataFrame, hpoa_path: str) -> pd.DataFrame:
    omim_to_g2p: dict[str, list[str]] = defaultdict(list)
    for gid, mim in zip(dd["g2p id"], dd["disease mim"]):
        for m in _pmids(mim):  # disease mim is the OMIM number
            omim_to_g2p[m].append(gid)
    cols = ["database_id", "disease_name", "qualifier", "hpo_id", "reference"]
    df = pd.read_csv(hpoa_path, sep="\t", comment="#", usecols=range(5), names=cols, header=0, dtype=str)
    rows = []
    for db, ref in zip(df["database_id"].astype(str), df["reference"].astype(str)):
        if not db.startswith("OMIM:"):
            continue
        gids = omim_to_g2p.get(db.split(":", 1)[1].strip(), [])
        for pmid in PMID_REF_RE.findall(ref):  # a cell may list several refs
            for gid in gids:
                rows.append((gid, pmid))
    return _frame(rows, "hpoa")


def clingen_caselevel(dd: pd.DataFrame, exports_dir: str) -> pd.DataFrame:
    """Case-level (genetic) ClinGen evidence PMIDs, matched to DDG2P by MONDO.

    `dd` is already restricted to leaf MONDOs, so a matched MONDO is a single gene-disease
    (a few leaf MONDOs still carry 2 G2P entries — mono/bi-allelic pairs — which we keep)."""
    mondo_to_g2p: dict[str, list[str]] = defaultdict(list)
    for gid, mondo in zip(dd["g2p id"], dd["disease MONDO"]):
        if mondo.strip():
            mondo_to_g2p[mondo.strip()].append(gid)
    rows = []
    for d in glob.glob(os.path.join(exports_dir, "HGNC_*_MONDO_*")):
        m = re.search(r"MONDO_(\d+)$", os.path.basename(d))
        gids = mondo_to_g2p.get(f"MONDO:{m.group(1)}", []) if m else []
        if not gids:
            continue
        pmids: set[str] = set()
        for f in glob.glob(os.path.join(d, "genetic_evidence_*tableExport.csv")):
            try:
                gdf = pd.read_csv(f, dtype=str).fillna("")
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            # PMIDs from the evidence Reference column only (not Explanation free-text)
            for c in (c for c in gdf.columns if "reference" in c.lower() and "pmid" in c.lower()):
                for v in gdf[c]:
                    pmids |= set(PMID_REF_RE.findall(str(v)))
        rows.extend((gid, p) for p in pmids for gid in gids)
    return _frame(rows, "clingen")


def _frame(rows, source: str) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=["key", "pmid"]).drop_duplicates()
    out.insert(0, "match_type", "g2p_id")
    out.insert(0, "source", source)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ddg2p", required=True, help="August-2025 DDG2P CSV (publications, disease mim, disease MONDO)")
    ap.add_argument("--mondo_backfill", default=None, help="Newer DDG2P CSV — MONDO backfill by g2p-id only")
    ap.add_argument("--hpoa", required=True, help="phenotype.hpoa")
    ap.add_argument("--clingen_exports", required=True,
                    help="ClinGen export dir of HGNC_*_MONDO_* subdirs (genetic_evidence_* tables)")
    ap.add_argument("--mondo_json", required=True,
                    help="MONDO obographs JSON (purl.obolibrary.org/obo/mondo.json) — for the leaf filter")
    ap.add_argument("--exclude_pmids", default=None,
                    help="CSV with a pmid column (e.g. annotated train/test set) to exclude from all truth sets")
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    dd = load_ddg2p(args.ddg2p, args.mondo_backfill)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dd, excl = restrict_to_leaf(dd, leaf_mondos(args.mondo_json))
    excl.to_csv(out / "excluded_grouping_mondos.csv", index=False)
    n_grp = int((excl["reason"] == "grouping_mondo").sum())
    print(f"[leaf] kept {len(dd)} leaf-MONDO entries; excluded {len(excl)} "
          f"({n_grp} grouping MONDOs, {len(excl) - n_grp} no-MONDO) -> excluded_grouping_mondos.csv")

    ts = pd.concat([premined(dd), hpoa(dd, args.hpoa), clingen_caselevel(dd, args.clingen_exports)],
                   ignore_index=True)

    if args.exclude_pmids:
        excl = set(pd.read_csv(args.exclude_pmids, dtype=str)["pmid"].astype(str))
        before = len(ts)
        ts = ts[~ts["pmid"].isin(excl)]
        print(f"[exclude_pmids] dropped {before - len(ts)} train/test pairs ({len(excl)} PMIDs excluded)")

    ts.to_csv(out / "truthsets.csv", index=False)
    print(f"\nWrote {len(ts):,} (key, pmid) ground-truth pairs -> {out}/truthsets.csv")
    for src, g in ts.groupby("source"):
        print(f"  {src:9s}: {len(g):>7,} pairs | {g['key'].nunique():>5,} keys | {g['pmid'].nunique():>7,} unique PMIDs")


if __name__ == "__main__":
    main()
