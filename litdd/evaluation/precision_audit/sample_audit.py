#!/usr/bin/env python3
"""Draw a blinded, stratified sample of the released map for a manual precision audit.

Each audited unit is one (PMID, assigned G2P id) mapping from the adjudication output
(one row per abstract with ``llm_dis_map`` and ``candidate_text``). Strata recorded per unit:

  recency            publication year band (``pubdate``)
  disease_volume     number of corpus mappings for that G2P id (rare, mid, high terciles)
  gene_multiplicity  single or multiple panel entries for the assigned gene

Allocation is equal per cell over the primary strata with a floor, so small cells are
oversampled; every stratum is recorded so precision and Wilson intervals can be computed per
stratum afterwards (``score_audit.py``). A minimum number of records published at or after
``--cutoff_year`` is guaranteed so precision can be compared before and after the
adjudication model's knowledge cutoff.

Outputs under ``--out_dir``: ``audit_worksheet.csv`` (blinded, annotator A),
``audit_worksheet_overlap.csv`` (the overlap subset for annotator B) and ``audit_key.csv``
(strata and assigned ids, not shown to annotators).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from litdd.evaluation.common import G2P_ID_RE, g2p_ids_split

RECENCY_BINS = [(0, 2009, "<=2009"), (2010, 2019, "2010-2019"), (2020, 9999, ">=2020")]
VERDICT_HELP = "correct | incorrect | uncertain"
ERROR_CATS = ("wrong_gene", "wrong_allelic_requirement", "wrong_mechanism", "somatic_only",
              "non_human_only", "acronym_gene_confusion", "cnv_snv_confusion",
              "no_molecular_confirmation", "wrong_disease_same_gene", "other")
STRATA = ["recency", "disease_volume", "gene_multiplicity"]


def recency_bin(year) -> str:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "year_unknown"
    for lo, hi, name in RECENCY_BINS:
        if lo <= y <= hi:
            return name
    return "year_unknown"


def _block_for(g2p_id: str, candidate_text) -> str:
    """The candidate text block whose id matches the assigned id, or ''."""
    if candidate_text is None:
        return ""
    for t in list(candidate_text):
        m = G2P_ID_RE.search(str(t))
        if m and m.group(0) == g2p_id:
            return str(t)
    return ""


def _field(block: str, label: str) -> str:
    """Value of a ``Label: value`` line in a candidate block, or ''."""
    for line in block.splitlines():
        if line.strip().lower().startswith(label.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def explode_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pmid, assigned g2p_id) with the candidate block, disease and gene."""
    rows = []
    for r in df.itertuples(index=False):
        for gid in sorted(g2p_ids_split(r.llm_dis_map)):
            if not gid.upper().startswith("G2P"):
                continue
            block = _block_for(gid, getattr(r, "candidate_text", None))
            rows.append({
                "pmid": int(r.pmid),
                "assigned_g2p_id": gid,
                "assigned_gene": _field(block, "Gene Symbol"),
                "assigned_disease": _field(block, "Disease Name"),
                "assigned_candidate_text": block,
                "year": r.pubdate,
                "title": r.title or "",
                "abstract": r.abstract or "",
            })
    return pd.DataFrame(rows)


def add_strata(units: pd.DataFrame, g2p_file: str) -> pd.DataFrame:
    units = units.copy()
    units["recency"] = units["year"].map(recency_bin)

    counts = units["assigned_g2p_id"].value_counts()
    units["_vol"] = units["assigned_g2p_id"].map(counts)
    try:
        units["disease_volume"] = pd.qcut(units["_vol"], 3, labels=["rare", "mid", "high"], duplicates="drop")
    except ValueError:
        units["disease_volume"] = "all"
    units["disease_volume"] = units["disease_volume"].astype(str)

    g2p = pd.read_csv(g2p_file, dtype=str, keep_default_na=False)
    g2p.columns = [c.strip() for c in g2p.columns]
    entries_per_gene = g2p.groupby("gene symbol").size()
    id_to_gene = dict(zip(g2p["g2p id"], g2p["gene symbol"]))
    gene = units["assigned_gene"].where(units["assigned_gene"] != "",
                                        units["assigned_g2p_id"].map(id_to_gene))
    units["assigned_gene"] = gene.fillna("")
    units["_n_entries"] = units["assigned_gene"].map(entries_per_gene).fillna(1).astype(int)
    units["gene_multiplicity"] = np.where(units["_n_entries"] > 1, "multiple", "single")
    return units.drop(columns=["_vol", "_n_entries"])


def stratified_sample(units: pd.DataFrame, n: int, primary_cols, floor: int, rng) -> pd.DataFrame:
    """Equal-per-cell allocation over the primary strata with a floor, random within cell."""
    cells = list(units.groupby(list(primary_cols)))
    base = max(floor, n // max(1, len(cells)))
    picked = []
    for _, cell in cells:
        take = min(base, len(cell))
        picked.append(cell.sample(n=take, random_state=rng.integers(1 << 31)))
    out = pd.concat(picked) if picked else units.iloc[:0]

    if len(out) > n:
        out = out.sample(n=n, random_state=rng.integers(1 << 31))
    elif len(out) < n:
        remainder = units.drop(index=out.index)
        extra = remainder.sample(n=min(n - len(out), len(remainder)), random_state=rng.integers(1 << 31))
        out = pd.concat([out, extra])
    return out.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)


def ensure_min_post_cutoff(sample: pd.DataFrame, units: pd.DataFrame, cutoff_year: int,
                           min_post: int, rng) -> pd.DataFrame:
    """Swap pre-cutoff units for post-cutoff ones until at least ``min_post`` records are
    published at or after ``cutoff_year``; the sample size is unchanged."""
    key = ["pmid", "assigned_g2p_id"]
    s_yr = pd.to_numeric(sample["year"], errors="coerce")
    need = min_post - int((s_yr >= cutoff_year).sum())
    if need <= 0:
        return sample
    sampled = set(map(tuple, sample[key].values))
    pool = units[pd.to_numeric(units["year"], errors="coerce") >= cutoff_year]
    pool = pool[~pool[key].apply(tuple, axis=1).isin(sampled)]
    if pool.empty:
        return sample
    add = pool.sample(n=min(need, len(pool)), random_state=rng.integers(1 << 31))
    pre = sample[s_yr < cutoff_year]
    drop = pre.sample(n=min(len(add), len(pre)), random_state=rng.integers(1 << 31))
    sample = pd.concat([sample.drop(index=drop.index), add], ignore_index=True)
    return sample.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="adjudication parquet with pmid, title, abstract, pubdate, candidate_text, llm_dis_map")
    ap.add_argument("--g2p_file", required=True, help="G2P export, for the gene-multiplicity stratum")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=500, help="audit sample size")
    ap.add_argument("--overlap", type=int, default=100, help="overlap subset size for inter-annotator kappa")
    ap.add_argument("--floor", type=int, default=40, help="minimum units per primary cell")
    ap.add_argument("--primary", nargs="+", default=["gene_multiplicity", "recency"],
                    help="strata used for allocation")
    ap.add_argument("--cutoff_year", type=int, default=2024,
                    help="year from which records count as published after the adjudication "
                         "model's knowledge cutoff")
    ap.add_argument("--min_post_cutoff", type=int, default=80,
                    help="minimum number of post-cutoff records in the sample")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cols = ["pmid", "title", "abstract", "pubdate", "candidate_text", "llm_dis_map"]
    df = pd.read_parquet(args.input, columns=cols)
    units = explode_mappings(df)
    units = add_strata(units, args.g2p_file)
    print(f"Corpus mappings: {len(units)} (from {df['pmid'].nunique()} PMIDs)")

    sample = stratified_sample(units, args.n, args.primary, args.floor, rng)
    if args.min_post_cutoff:
        sample = ensure_min_post_cutoff(sample, units, args.cutoff_year, args.min_post_cutoff, rng)
    n_post = int((pd.to_numeric(sample["year"], errors="coerce") >= args.cutoff_year).sum())
    sample.insert(0, "audit_id", [f"A{i:04d}" for i in range(len(sample))])
    overlap_ids = set(sample["audit_id"].sample(n=min(args.overlap, len(sample)),
                                                random_state=rng.integers(1 << 31)))
    sample["in_overlap"] = sample["audit_id"].isin(overlap_ids)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    blind_cols = ["audit_id", "pmid", "title", "abstract", "assigned_g2p_id",
                  "assigned_disease", "assigned_gene", "assigned_candidate_text"]
    worksheet = sample[blind_cols].copy()
    worksheet["verdict"] = ""
    worksheet["error_category"] = ""
    worksheet["notes"] = ""
    worksheet.to_csv(out / "audit_worksheet.csv", index=False)
    worksheet[sample["in_overlap"].values].to_csv(out / "audit_worksheet_overlap.csv", index=False)

    key_cols = ["audit_id", "pmid", "assigned_g2p_id", "year", *STRATA, "in_overlap"]
    sample[key_cols].to_csv(out / "audit_key.csv", index=False)

    print(f"Wrote {len(worksheet)} units to {out}/audit_worksheet.csv "
          f"({len(overlap_ids)} in overlap; {n_post} published >= {args.cutoff_year})")
    print("Verdict values:", VERDICT_HELP, "| error categories:", ", ".join(ERROR_CATS))
    print("\nStratum coverage:")
    for col in STRATA:
        print(f"  {col}:", dict(sample[col].value_counts()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
