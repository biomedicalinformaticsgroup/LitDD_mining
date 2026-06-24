#!/usr/bin/env python3
"""Draw a blinded, stratified sample of the deployed LitDD corpus for a manual
precision audit (Reviewer 2 B1 / R2-P1,P2).

The deployed corpus is the set of (PMID -> G2P disease) mappings the pipeline emits
(e.g. ``final_tiab_mappings.parquet``). Its precision was never measured directly — the
headline 0.83 is from a ~26%-positive balanced test set, which does not transfer to the
deployed base rate. This script stratifies the corpus and draws a sample for a clinical
geneticist to label as correct / incorrect, **blinded** to the model score and stratum.

Each audited unit is one (PMID, assigned G2P id) mapping. Strata tagged per unit:
  - confidence : cross-encoder score of the assigned id (from ``top5_cross``)
  - recency    : publication year (``pubdate``)
  - disease_volume : number of corpus mappings for that G2P id (rare/mid/high terciles)
  - gene_multiplicity : single vs multiple DDG2P entries for the assigned gene

The primary allocation is equal-per-cell over (confidence x gene_multiplicity) with a
floor, which oversamples the small high-uncertainty cells; all four strata are recorded
so precision + Wilson CIs can be computed per stratum afterwards (see score_audit.py).

Outputs (default under revision/precision_audit/, which is gitignored — the worksheet
contains real abstracts and is annotator-facing, not committed):
  audit_worksheet.csv          blinded; annotator A labels the full sample
  audit_worksheet_overlap.csv  the overlap subset; annotator B labels this for kappa
  audit_key.csv                audit_id -> score/strata/assigned id (NOT shown to annotators)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LGMDE_DISEASE_FIELD = 5  # g2p_id - gene - gene_mim - hgnc - prev_symbols - DISEASE NAME - ...
CONF_BINS = [(0.90, 0.95), (0.95, 0.99), (0.99, 1.01)]
RECENCY_BINS = [(0, 2009, "<=2009"), (2010, 2019, "2010-2019"), (2020, 9999, ">=2020")]
VERDICT_HELP = "correct | incorrect | uncertain"
ERROR_CATS = ("wrong_gene", "wrong_allelic_requirement", "wrong_mechanism", "somatic_only",
              "non_human_only", "acronym_gene_confusion", "cnv_snv_confusion",
              "no_molecular_confirmation", "wrong_disease_same_gene", "other")


def conf_bin(score: float) -> str:
    for lo, hi in CONF_BINS:
        if lo <= score < hi:
            return f"conf_{lo:.2f}-{hi:.2f}".replace("1.01", "1.00")
    return "conf_other"


def recency_bin(year) -> str:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "year_unknown"
    for lo, hi, name in RECENCY_BINS:
        if lo <= y <= hi:
            return name
    return "year_unknown"


def _thread_for(g2p_id: str, threads) -> str:
    """Find the candidate LGMDE thread string whose id matches the assigned id."""
    if threads is None:
        return ""
    for t in threads:
        if isinstance(t, str) and t.split(" - ", 1)[0].strip() == g2p_id:
            return t
    return ""


def _score_for(g2p_id: str, top5) -> float:
    """Cross-encoder score of the assigned id from the top5_cross list of {label, score}."""
    if top5 is None:
        return float("nan")
    for c in top5:
        label = c.get("label", "") if isinstance(c, dict) else ""
        if label.split(" - ", 1)[0].strip() == g2p_id:
            return float(c.get("score", float("nan")))
    return float("nan")


def explode_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (pmid, assigned g2p_id) with score, thread, disease, gene."""
    rows = []
    for r in df.itertuples(index=False):
        ans = r.llm_dis_map
        if not isinstance(ans, str) or not ans.strip() or ans.strip().upper() == "NO MATCH":
            continue
        for gid in [x.strip() for x in ans.split(";") if x.strip().upper().startswith("G2P")]:
            thread = _thread_for(gid, r.top_5_cross_lgmde)
            parts = [p.strip() for p in thread.split(" - ")] if thread else []
            rows.append({
                "pmid": int(r.pmid),
                "assigned_g2p_id": gid,
                "assigned_gene": parts[1] if len(parts) > 1 else "",
                "assigned_disease": parts[LGMDE_DISEASE_FIELD] if len(parts) > LGMDE_DISEASE_FIELD else "",
                "assigned_lgmde_thread": thread,
                "score": _score_for(gid, r.top5_cross),
                "year": r.pubdate,
                "title": r.title or "",
                "abstract": r.abstract or "",
            })
    return pd.DataFrame(rows)


def add_strata(units: pd.DataFrame, g2p_file: str) -> pd.DataFrame:
    units = units.copy()
    units["confidence"] = units["score"].map(conf_bin)
    units["recency"] = units["year"].map(recency_bin)

    # disease_volume: corpus mappings per assigned id -> terciles
    counts = units["assigned_g2p_id"].value_counts()
    units["_vol"] = units["assigned_g2p_id"].map(counts)
    try:
        units["disease_volume"] = pd.qcut(units["_vol"], 3, labels=["rare", "mid", "high"], duplicates="drop")
    except ValueError:
        units["disease_volume"] = "all"
    units["disease_volume"] = units["disease_volume"].astype(str)

    # gene_multiplicity: # DDG2P entries per gene symbol in the G2P CSV
    g2p = pd.read_csv(g2p_file, dtype=str, keep_default_na=False)
    gene_col = "gene symbol" if "gene symbol" in g2p.columns else g2p.columns[1]
    entries_per_gene = g2p.groupby(gene_col).size()
    units["_n_ddg2p"] = units["assigned_gene"].map(entries_per_gene).fillna(1).astype(int)
    units["gene_multiplicity"] = np.where(units["_n_ddg2p"] > 1, "multiple", "single")
    return units.drop(columns=["_vol", "_n_ddg2p"])


def stratified_sample(units: pd.DataFrame, n: int, primary_cols, floor: int, rng) -> pd.DataFrame:
    """Equal-per-cell allocation over the primary strata (with a floor), random within cell."""
    cells = list(units.groupby(list(primary_cols)))
    base = max(floor, n // max(1, len(cells)))
    picked = []
    for _, cell in cells:
        take = min(base, len(cell))
        picked.append(cell.sample(n=take, random_state=rng.integers(1 << 31)))
    out = pd.concat(picked) if picked else units.iloc[:0]

    # Trim to ~n (drop from the largest cells) or top up from the remainder.
    if len(out) > n:
        out = out.sample(n=n, random_state=rng.integers(1 << 31))
    elif len(out) < n:
        remainder = units.drop(index=out.index)
        extra = remainder.sample(n=min(n - len(out), len(remainder)), random_state=rng.integers(1 << 31))
        out = pd.concat([out, extra])
    return out.sample(frac=1, random_state=rng.integers(1 << 31)).reset_index(drop=True)


def ensure_min_post_cutoff(sample: pd.DataFrame, units: pd.DataFrame, cutoff_year: int,
                           min_post: int, rng) -> pd.DataFrame:
    """Guarantee at least `min_post` records published at/after `cutoff_year` (for an
    adequately-powered post-cutoff / LLM-contamination check, R3.1), swapping in extra
    post-cutoff units for pre-cutoff ones to keep the sample size constant."""
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


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Deployed-corpus parquet (e.g. final_tiab_mappings.parquet)")
    ap.add_argument("--g2p_file", required=True, help="G2P DD CSV (for gene-multiplicity strata)")
    ap.add_argument("--out_dir", default="revision/precision_audit", help="Output dir (gitignored)")
    ap.add_argument("--n", type=int, default=500, help="Audit sample size")
    ap.add_argument("--overlap", type=int, default=100, help="Overlap subset size for inter-annotator kappa")
    ap.add_argument("--floor", type=int, default=40, help="Minimum units per primary cell")
    ap.add_argument("--primary", nargs="+", default=["confidence", "gene_multiplicity"],
                    help="Strata used for allocation (oversampled)")
    ap.add_argument("--cutoff_year", type=int, default=2024,
                    help="LLM knowledge-cutoff year; guarantee --min_post_cutoff records at/after it (R3.1)")
    ap.add_argument("--min_post_cutoff", type=int, default=80,
                    help="Minimum post-cutoff records for an adequately-powered contamination check")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cols = ["pmid", "title", "abstract", "pubdate", "top5_cross", "top_5_cross_lgmde", "llm_dis_map"]
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
                  "assigned_disease", "assigned_gene", "assigned_lgmde_thread"]
    worksheet = sample[blind_cols].copy()
    worksheet["verdict"] = ""          # annotator fills: correct | incorrect | uncertain
    worksheet["error_category"] = ""   # if incorrect, one of ERROR_CATS
    worksheet["notes"] = ""
    worksheet.to_csv(out / "audit_worksheet.csv", index=False)
    worksheet[sample["in_overlap"].values].to_csv(out / "audit_worksheet_overlap.csv", index=False)

    key_cols = ["audit_id", "pmid", "assigned_g2p_id", "score", "confidence",
                "year", "recency", "disease_volume", "gene_multiplicity", "in_overlap"]
    sample[key_cols].to_csv(out / "audit_key.csv", index=False)

    print(f"Wrote {len(worksheet)} units to {out}/audit_worksheet.csv "
          f"({len(overlap_ids)} in overlap; {n_post} post-cutoff >= {args.cutoff_year})")
    print("Verdict values:", VERDICT_HELP, "| error categories:", ", ".join(ERROR_CATS))
    print("\nStratum coverage:")
    for col in ["confidence", "recency", "disease_volume", "gene_multiplicity"]:
        print(f"  {col}:", dict(sample[col].value_counts()))


if __name__ == "__main__":
    main()
