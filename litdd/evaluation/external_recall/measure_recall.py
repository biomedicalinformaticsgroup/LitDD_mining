#!/usr/bin/env python3
"""Measure PMID-retrieval recall per disease (G2P id) against the external truth sets.

For each disease the ground truth is the set of curated PMIDs from HPOA, ClinGen and the
pre-mined DDG2P publications for that entry, and the mined set is the PMIDs the released map
assigns to it. Per disease, recall = |mined and truth| / |truth|:

  micro recall = sum over diseases of |mined and truth| / sum over diseases of |truth|
  macro recall = mean over diseases of |mined and truth| / |truth|

Each miss is categorised from the complete pipeline parquet (one row per screen-positive
abstract with ``llm_dis_map``): ``litdd_bert_negative`` (the screen rejected the paper),
``llm_no_match`` (the adjudicator returned NO MATCH), ``mapped_other`` (mapped to another
entry) or ``not_in_final_map`` (mapped to the entry but absent from the released map).

Inputs: ``truthsets.csv`` from ``build_truthsets.py`` (columns ``source``, ``key``, ``pmid``),
the released map CSV (``pmid``, ``g2p_id``) and the complete pipeline parquet. Writes
``recall_summary.csv`` and ``miss_categories.csv`` to ``--out_dir``.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from litdd.evaluation.common import g2p_ids_split

REPORTABLE = ("premined", "hpoa", "clingen")


def load_pipeline_state(complete_df: str) -> tuple[dict[str, set[str]], dict[str, int]]:
    """``pmid -> set of mapped g2p ids`` and ``pmid -> publication year`` over every
    screen-positive abstract in the complete pipeline parquet."""
    df = pd.read_parquet(complete_df, columns=["pmid", "pubdate", "llm_dis_map"])
    state: dict[str, set[str]] = {}
    years: dict[str, int] = {}
    for pmid, pub, ans in zip(df["pmid"].astype("int64"), df["pubdate"], df["llm_dis_map"]):
        try:
            years[str(pmid)] = int(pub)
        except (TypeError, ValueError):
            pass
        d = state.setdefault(str(pmid), set())
        d |= {g for g in g2p_ids_split(ans) if g.upper().startswith("G2P")}
    return state, years


def mined_deployed(litdd_map: str) -> dict[str, set[str]]:
    """``g2p_id -> set of pmids`` from the released map CSV; a cell may hold several ids."""
    m = pd.read_csv(litdd_map, dtype=str).fillna("")
    col = "g2p_id" if "g2p_id" in m.columns else m.columns[1]
    out: dict[str, set[str]] = defaultdict(set)
    for p, g in zip(m["pmid"], m[col]):
        for gid in g.split(";"):
            if gid.strip():
                out[gid.strip()].add(str(p))
    return out


def recall_stats(truth: dict[str, set[str]], mined: dict[str, set[str]], restrict=None):
    """Micro and macro recall, number of diseases and number of truth PMIDs counted.
    ``restrict`` limits the truth PMIDs to those in the given set."""
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


def classify_miss(g2p: str, pmid: str, state: dict[str, set[str]]) -> str:
    if pmid not in state:
        return "litdd_bert_negative"
    mapped = state[pmid]
    if not mapped:
        return "llm_no_match"
    if g2p not in mapped:
        return "mapped_other"
    return "not_in_final_map"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truthsets", required=True, help="truthsets.csv from build_truthsets.py")
    ap.add_argument("--litdd_map", required=True, help="released (pmid, g2p_id) map CSV")
    ap.add_argument("--complete_df", required=True, help="complete pipeline parquet")
    ap.add_argument("--pmid_years", default=None,
                    help="CSV (pmid, year) for truth PMIDs absent from the pipeline parquet (from fetch_pmid_meta.py)")
    ap.add_argument("--min_year", type=int, default=None,
                    help="exclude truth PMIDs published before this year")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> int:
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
        print(f"min_year={args.min_year}: dropped {before - len(ts)} earlier truth pairs")

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

    screen_positive = set(state)
    mined = mined_deployed(args.litdd_map)

    rows = []
    for src, truth in truth_by_src.items():
        for scope, restrict in (("all", None), ("bert_positive", screen_positive)):
            micro, macro, n_dis, n_pmid = recall_stats(truth, mined, restrict)
            rows.append({"source": src, "reportable": src in REPORTABLE or src == "combined",
                         "scope": scope, "n_diseases": n_dis, "n_truth_pmids": n_pmid,
                         "micro_recall": round(micro, 3), "macro_recall": round(macro, 3)})
    summary = pd.DataFrame(rows)

    misses = []
    for src, truth in truth_by_src.items():
        if src == "combined":
            continue
        for g, ps in truth.items():
            for p in ps - mined.get(g, set()):
                misses.append((src, classify_miss(g, p, state)))
    miss = (pd.DataFrame(misses, columns=["source", "category"]).value_counts()
            .rename("n").reset_index())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "recall_summary.csv", index=False)
    miss.to_csv(out / "miss_categories.csv", index=False)

    print("=== Recall on external sets, per disease (G2P id), micro and macro ===")
    print(summary.to_string(index=False))
    print("\n=== Miss categories ===")
    print(miss.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
