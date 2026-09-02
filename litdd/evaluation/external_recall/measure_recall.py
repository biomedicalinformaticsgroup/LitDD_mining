#!/usr/bin/env python3
"""Measure LitDD PMID-retrieval recall against the external ground-truth sets, per disease
(G2P ID), reported as micro and macro recall (Reviewer 3 R3.4 / Reviewer 2 C1/C2).

For each disease (G2P ID) the ground truth is the set of curated PMIDs from
HPOA / ClinGen / pre-mined DDG2P for that entry. The "mined entry" is the set of PMIDs
LitDD mapped to that G2P ID. Per disease, recall = |mined ∩ truth| / |truth|.
  micro recall = sum_d |mined ∩ truth| / sum_d |truth|     (pooled over PMIDs)
  macro recall = mean_d ( |mined ∩ truth| / |truth| )       (mean over diseases)

Two corpus variants:
  deployed : the shipped (PMID -> G2P) map (score>=cutoff AND gene-mention filter)
  relaxed  : score>=cutoff only, gene-mention filter OFF (from the complete pipeline df)
(A third, API-gene, slots in once the PubTator3 fetch lands.)

Each deployed miss is categorised (litdd_bert_negative / llm_no_match / mapped_other /
below_score / gene_filtered) so a miss is explained rather than asserted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

CUT = 0.9
# Sources reported in the manuscript. ClinGen uses CASE-LEVEL (genetic) evidence only —
# the experimental/functional ClinGen evidence (mouse/zebrafish/mechanism) is excluded at
# truth-set build, matching the manuscript's "case level evidence" definition (Table 6).
REPORTABLE = ("premined", "hpoa", "clingen")


def load_pipeline_state(complete_df: str):
    """pmid -> {g2p_id: best_score}, and pmid -> year, over all BERT-positive abstracts."""
    df = pd.read_parquet(complete_df, columns=["pmid", "pubdate", "llm_dis_map", "top5_cross"])
    state: dict[str, dict[str, float]] = {}
    years: dict[str, int] = {}
    for pmid, pub, ans, top5 in zip(df["pmid"].astype("int64"), df["pubdate"],
                                    df["llm_dis_map"], df["top5_cross"]):
        try:
            years[str(pmid)] = int(pub)
        except (TypeError, ValueError):
            pass
        d = state.setdefault(str(pmid), {})
        a = str(ans)
        if a.strip().upper() in ("", "NO MATCH", "NAN"):
            continue
        scores = {}
        if top5 is not None:
            for c in top5:
                lab = c.get("label", "") if isinstance(c, dict) else ""
                g = lab.split(" - ", 1)[0].strip()
                if g:
                    scores[g] = float(c.get("score", 0.0))
        for g in (x.strip() for x in a.split(";")):
            if g.upper().startswith("G2P"):
                d[g] = max(d.get(g, 0.0), scores.get(g, 0.0))
    return state, years


def mined_deployed(litdd_map: str) -> dict[str, set[str]]:
    m = pd.read_csv(litdd_map, dtype=str).fillna("")
    col = "g2p_id" if "g2p_id" in m.columns else m.columns[1]
    out: dict[str, set[str]] = defaultdict(set)
    for p, g in zip(m["pmid"], m[col]):
        for gid in g.split(";"):  # a PMID can map to several G2P IDs (8% of rows)
            if gid.strip():
                out[gid.strip()].add(str(p))
    return out


def mined_relaxed(state, cutoff) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for pmid, mp in state.items():
        for g, s in mp.items():
            if s >= cutoff:
                out[g].add(pmid)
    return out


def recall_stats(truth: dict[str, set[str]], mined: dict[str, set[str]], restrict=None):
    """restrict: if given, only count truth PMIDs in this set (e.g. PMIDs in the corpus)."""
    inter = total = 0
    per_disease = []
    for g, tp in truth.items():
        if restrict is not None:
            tp = tp & restrict
        if not tp:
            continue
        hit = len(tp & mined.get(g, set()))
        inter += hit
        total += len(tp)
        per_disease.append(hit / len(tp))
    micro = inter / total if total else 0.0
    macro = sum(per_disease) / len(per_disease) if per_disease else 0.0
    return micro, macro, len(per_disease), total


def classify_miss(g2p, pmid, state, cutoff) -> str:
    if pmid not in state:
        return "litdd_bert_negative"  # LitDD's BERT classified the PMID negative (over all PubMed)
    mapped = state[pmid]
    if not mapped:
        return "llm_no_match"
    if g2p not in mapped:
        return "mapped_other"
    return "below_score" if mapped[g2p] < cutoff else "gene_filtered"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", default="revision/external_recall/truthsets.csv")
    ap.add_argument("--litdd_map", required=True, help="Deployed (PMID,G2P) map CSV")
    ap.add_argument("--complete_df", required=True, help="pipeline_df_complete.parquet")
    ap.add_argument("--score_cutoff", type=float, default=CUT)
    ap.add_argument("--pmid_years", default=None,
                    help="CSV (pmid,year) for BERT-negative truth PMIDs (from fetch_pmid_meta.py)")
    ap.add_argument("--min_year", type=int, default=None,
                    help="Exclude truth PMIDs published before this year (LitDD filters pubdate>1980; use 1981)")
    ap.add_argument("--out_dir", default="revision/external_recall")
    return ap.parse_args()


def main():
    args = parse_args()
    ts = pd.read_csv(args.truthsets, dtype=str)

    state, years = load_pipeline_state(args.complete_df)
    if args.pmid_years:
        ey = pd.read_csv(args.pmid_years, dtype=str)
        for p, y in zip(ey["pmid"], pd.to_numeric(ey["year"], errors="coerce")):
            if y == y and p not in years:
                years[p] = int(y)

    if args.min_year:
        before = len(ts)
        ts = ts[ts["pmid"].map(lambda p: years.get(p, 9999) >= args.min_year)]
        print(f"[min_year={args.min_year}] dropped {before - len(ts)} pre-{args.min_year} truth pairs")

    # truth per source as g2p_id -> set(pmids); combined over REPORTABLE sources only
    truth_by_src: dict[str, dict[str, set[str]]] = {}
    for src, g in ts.groupby("source"):
        d: dict[str, set[str]] = defaultdict(set)
        for k, p in zip(g["key"], g["pmid"]):
            d[k].add(p)
        truth_by_src[src] = d
    combined: dict[str, set[str]] = defaultdict(set)
    for src in REPORTABLE:
        for g, ps in truth_by_src.get(src, {}).items():
            combined[g] |= ps
    truth_by_src["combined"] = combined

    bert_positive_pmids = set(state)  # PMIDs LitDD's BERT classified positive
    variants = {"deployed": mined_deployed(args.litdd_map),
                "relaxed": mined_relaxed(state, args.score_cutoff)}

    rows = []
    for src, truth in truth_by_src.items():
        for vname, mined in variants.items():
            # scope=all : all curated PMIDs; scope=bert_positive : only curated PMIDs LitDD's
            # BERT classified positive (excludes BERT-negative papers, e.g. no molecular confirmation)
            for scope, restrict in (("all", None), ("bert_positive", bert_positive_pmids)):
                micro, macro, n_dis, n_pmid = recall_stats(truth, mined, restrict)
                rows.append({"source": src, "reportable": src in REPORTABLE or src == "combined",
                             "variant": vname, "scope": scope, "n_diseases": n_dis,
                             "n_truth_pmids": n_pmid, "micro_recall": round(micro, 3),
                             "macro_recall": round(macro, 3)})
    summary = pd.DataFrame(rows)

    # miss categories on the deployed variant (per source, excl. combined)
    misses = []
    dep = variants["deployed"]
    for src, truth in truth_by_src.items():
        if src == "combined":
            continue
        for g, ps in truth.items():
            for p in ps - dep.get(g, set()):
                misses.append((src, classify_miss(g, p, state, args.score_cutoff)))
    miss = (pd.DataFrame(misses, columns=["source", "category"]).value_counts()
            .rename("n").reset_index())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "recall_summary.csv", index=False)
    miss.to_csv(out / "miss_categories.csv", index=False)

    print("=== LitDD recall on external sets — per disease (G2P ID), micro & macro (R3.4) ===")
    print(summary.to_string(index=False))
    print("\n=== Deployed miss categories ===")
    print(miss.to_string(index=False))


if __name__ == "__main__":
    main()
