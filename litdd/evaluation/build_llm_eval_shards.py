#!/usr/bin/env python3
"""Build the annotated-set fixture for evaluating the LLM adjudication stage.

The annotated test split (``ds_test``: 2,779 (tiab, g2p_lgmde, label) pairs over 2,739
TIABs) is pair-level: a TIAB can carry several positive G2P entries and several labelled
negatives. The original paper scored the LLM stage on exactly this split with the deployed
cross-encoder's top-5 candidates attached (``ds_test_crossencoded.pkl``). This script turns
that into the ``llm_map.py`` contract plus a gold key:

  <out_dir>/shards/annotated_test.parquet   one row per TIAB: pmid, row_id, tiab,
                                            bert_predict [, top5_cross list<struct<label,score>>]
  <out_dir>/gold.csv                        pmid, row_id, true_g2p_ids (';'-joined, empty when
                                            the TIAB has no positive), n_gold, genereviews,
                                            bert_predict, gold_in_candidates, max_cross_score
  <out_dir>/pairs.csv                       every labelled (pmid, g2p_id, label) pair, for
                                            the pair-level metric view
  <out_dir>/provenance.json

Two candidate modes:
  --candidates paper   attach the pickle's deployed-cross-encoder top-5 (the paper's own
                       candidate set; requires the 2025-02-15 export, verified byte-for-byte)
  --candidates none    write TIABs only; candidates are produced downstream by the current
                       cross-encoder (full-panel top-k, or gene gate -> candidates), so the
                       fixture can be evaluated against the CURRENT G2P export

G2P VERSION. ``--g2p_csv`` must be the export the *arm* uses. Labels are resolved by G2P id;
with ``--drop_retired`` TIABs whose gold entry no longer exists in that export are dropped
and counted (8 of 654 for 2025-02-15 -> 2026-06-24), otherwise a retired gold id aborts.

``--screen_preds`` replaces the pickle's ``bert_predict`` (the originally deployed screen)
with another checkpoint's per-PMID predictions (csv: pmid,label,prob,pred), e.g. the released
add20k screen from ``revision/dstest_eval.py``.

Usage:
    REF=/path/to/ddg2p_pubmed2diseasemodel_CLEAN/clean_pipeline/train_test
    python litdd/evaluation/build_llm_eval_shards.py \\
        --crossencoded_pkl $REF/ds_test_crossencoded.pkl \\
        --anno_csv $REF/g2p_id_tiab_anno_df_FINAL.csv \\
        --g2p_csv $REF/G2P_DD_2025-02-15.csv --candidates paper \\
        --out_dir revision/llm_eval/annotated_2025
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from litdd.threads import build_lgmde_map  # noqa: E402

G2P_ID_RE = re.compile(r"^(G2P\d+)")
GENEREVIEWS_MARKER = "CLINICAL CHARACTERISTICS"


def thread_id(thread: str) -> str:
    m = G2P_ID_RE.match(str(thread).strip())
    if not m:
        raise ValueError(f"thread does not start with a G2P id: {thread[:80]!r}")
    return m.group(1)


def normalise_thread(s: str) -> str:
    """Only rendering artefact between the pickle and litdd.threads: 'HGNC:HGNC:' (fixed)."""
    return str(s).replace("HGNC:HGNC:", "HGNC:").rstrip()


def verify_panel(df: pd.DataFrame, panel: dict, g2p_csv: str, check_threads: bool) -> dict:
    """Gold ids must exist in the export; with check_threads every thread must re-render."""
    gold = df.loc[df["label"] == 1, "g2p_lgmde"].unique().tolist()
    gold_ids = {thread_id(g) for g in gold}
    report = {"g2p_csv": g2p_csv, "panel_entries": len(panel),
              "gold_ids": len(gold_ids), "gold_ids_in_panel": sum(i in panel for i in gold_ids),
              "gold_ids_retired": sorted(i for i in gold_ids if i not in panel)}
    if check_threads:
        cands = {lab for row in df["top_5_cross"] for lab, _ in row}
        cand_ok = sum(normalise_thread(panel.get(thread_id(c), "")) == normalise_thread(c)
                      for c in cands)
        gold_ok = sum(normalise_thread(panel.get(thread_id(g), "")) == normalise_thread(g)
                      for g in gold)
        report.update({"candidate_threads": len(cands), "candidate_threads_match_panel": cand_ok,
                       "gold_threads": len(gold), "gold_threads_match_panel": gold_ok})
        if cand_ok != len(cands) or gold_ok != len(gold):
            raise SystemExit(f"[ERROR] fixture threads do not re-render from {g2p_csv}: "
                             f"{report}. Wrong G2P export -- the labels and candidates belong "
                             "to another version.")
    return report


def build(df: pd.DataFrame, anno: pd.DataFrame, panel: dict, with_candidates: bool,
          drop_retired: bool, screen_preds: dict | None):
    df = df.copy()
    df["g2p_id"] = df["g2p_lgmde"].map(thread_id)
    # TIAB -> pmid (the split was made on tiab; pmid lives in the annotation frame)
    t2p = (anno.dropna(subset=["pmid"]).assign(pmid=lambda d: d["pmid"].astype(int).astype(str))
           .groupby("tiab")["pmid"].agg(lambda s: sorted(set(s))))
    rows, gold_rows, pair_rows, dropped = [], [], [], []
    for tiab, grp in df.groupby("tiab", sort=False):
        pmids = t2p.get(tiab, [])
        pmid = pmids[0] if pmids else None
        gold_ids = sorted(set(grp.loc[grp["label"] == 1, "g2p_id"]))
        row_id = f"pmid{pmid}" if pmid else f"tiab{len(rows) + len(dropped):05d}"
        retired = [g for g in gold_ids if g not in panel]
        if retired:
            if not drop_retired:
                raise SystemExit(f"[ERROR] gold id(s) {retired} for {row_id} are not in the "
                                 "export; pass --drop_retired or use the matching export")
            dropped.append({"row_id": row_id, "pmid": pmid, "retired_gold_ids": ";".join(retired)})
            continue
        bert = int(grp["bert_predict"].iloc[0])
        if screen_preds is not None:
            if pmid is None or pmid not in screen_preds:
                raise SystemExit(f"[ERROR] --screen_preds has no prediction for pmid {pmid}")
            bert = int(screen_preds[pmid])
        cand = list(grp["top_5_cross"].iloc[0]) if with_candidates else []
        cand_ids = [thread_id(str(lab)) for lab, _ in cand]
        row = {"pmid": pmid or row_id, "row_id": row_id, "tiab": tiab, "bert_predict": bert}
        if with_candidates:
            row["top5_cross"] = [{"label": str(lab), "score": float(sc)} for lab, sc in cand]
            row["n_candidates"] = len(cand)
        rows.append(row)
        gold_rows.append({
            "pmid": pmid or row_id, "row_id": row_id,
            "true_g2p_ids": ";".join(gold_ids), "n_gold": len(gold_ids),
            "n_labelled_pairs": len(grp),
            "genereviews": GENEREVIEWS_MARKER in tiab,
            "bert_predict": bert,
            "gold_in_candidates": sum(g in cand_ids for g in gold_ids) if with_candidates else None,
            "max_cross_score": max((float(sc) for _, sc in cand), default=float("nan"))
            if with_candidates else float("nan"),
            "n_pmids_for_tiab": len(pmids),
        })
        for _, r in grp.iterrows():
            pair_rows.append({"pmid": pmid or row_id, "row_id": row_id,
                              "g2p_id": r["g2p_id"], "label": int(r["label"]),
                              "in_candidates": (r["g2p_id"] in cand_ids) if with_candidates else None,
                              "in_panel": r["g2p_id"] in panel})
    return pd.DataFrame(rows), pd.DataFrame(gold_rows), pd.DataFrame(pair_rows), pd.DataFrame(dropped)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--crossencoded_pkl", required=True)
    ap.add_argument("--anno_csv", required=True, help="g2p_id_tiab_anno_df_FINAL.csv (pmid, tiab)")
    ap.add_argument("--g2p_csv", required=True, help="the export this arm of the evaluation uses")
    ap.add_argument("--candidates", choices=["paper", "none"], default="paper")
    ap.add_argument("--drop_retired", action="store_true",
                    help="drop TIABs whose gold entry is not in --g2p_csv (and record them)")
    ap.add_argument("--screen_preds", default=None,
                    help="csv (pmid,label,prob,pred) overriding bert_predict, e.g. the released screen")
    ap.add_argument("--corrections", default=None,
                    help="data/annotation_corrections.csv: relabel positive pairs (pmid, g2p_id_from "
                         "-> g2p_id_to) before building gold/pairs; the pickle's thread string is "
                         "re-rendered from --g2p_csv for the corrected entry")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = pd.read_pickle(args.crossencoded_pkl)
    need = {"tiab", "g2p_lgmde", "label", "bert_predict", "top_5_cross"}
    if not need <= set(df.columns):
        raise SystemExit(f"[ERROR] {args.crossencoded_pkl} lacks {need - set(df.columns)}")
    anno = pd.read_csv(args.anno_csv, usecols=["pmid", "tiab"])
    panel = build_lgmde_map(args.g2p_csv)
    n_corr = 0
    if args.corrections:
        corr = pd.read_csv(args.corrections, dtype=str)
        pm = (anno.dropna(subset=["pmid"]).assign(pmid=lambda d: d["pmid"].astype(int).astype(str))
              .drop_duplicates("tiab").set_index("tiab")["pmid"])
        df = df.copy()
        df["_pmid"] = df["tiab"].map(pm)
        df["_id"] = df["g2p_lgmde"].map(thread_id)
        for c in corr.itertuples(index=False):
            m = (df["_pmid"] == c.pmid) & (df["_id"] == c.g2p_id_from) & (df["label"] == 1)
            if m.any():
                if c.g2p_id_to not in panel:
                    raise SystemExit(f"[ERROR] correction target {c.g2p_id_to} not in {args.g2p_csv}")
                df.loc[m, "g2p_lgmde"] = panel[c.g2p_id_to]
                n_corr += int(m.sum())
        df = df.drop(columns=["_pmid", "_id"])
        print(f"[Info] applied {n_corr} label corrections from {args.corrections}")
    screen = None
    if args.screen_preds:
        sp = pd.read_csv(args.screen_preds, dtype={"pmid": str})
        screen = dict(zip(sp["pmid"].astype(str), sp["pred"].astype(int)))

    with_cands = args.candidates == "paper"
    panel_report = verify_panel(df, panel, args.g2p_csv, check_threads=with_cands)
    shard, gold, pairs, dropped = build(df, anno, panel, with_cands, args.drop_retired, screen)

    os.makedirs(os.path.join(args.out_dir, "shards"), exist_ok=True)
    shard_path = os.path.join(args.out_dir, "shards", "annotated_test.parquet")
    shard.to_parquet(shard_path, index=False)
    gold.to_csv(os.path.join(args.out_dir, "gold.csv"), index=False)
    pairs.to_csv(os.path.join(args.out_dir, "pairs.csv"), index=False)
    if len(dropped):
        dropped.to_csv(os.path.join(args.out_dir, "dropped_retired.csv"), index=False)

    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = None
    prov = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "litdd/evaluation/build_llm_eval_shards.py", "git_commit": sha,
        "crossencoded_pkl": os.path.abspath(args.crossencoded_pkl),
        "anno_csv": os.path.abspath(args.anno_csv),
        "candidates": args.candidates,
        "corrections": os.path.abspath(args.corrections) if args.corrections else None,
        "n_label_corrections_applied": n_corr,
        "screen_preds": os.path.abspath(args.screen_preds) if args.screen_preds else
        "bert_predict from the pickle (originally deployed screen)",
        "panel_check": panel_report,
        "n_pairs": int(len(df)), "n_tiabs": int(len(shard)),
        "n_tiabs_dropped_retired_gold": int(len(dropped)),
        "n_tiabs_with_gold": int((gold["n_gold"] > 0).sum()),
        "n_tiabs_multi_gold": int((gold["n_gold"] > 1).sum()),
        "n_gold_ids": int(gold["n_gold"].sum()),
        "n_screen_positive": int((gold["bert_predict"] == 1).sum()),
        "n_screen_positive_with_gold": int(((gold["bert_predict"] == 1) & (gold["n_gold"] > 0)).sum()),
        "n_genereviews": int(gold["genereviews"].sum()),
        "n_tiabs_without_pmid": int(gold["row_id"].str.startswith("tiab").sum()),
        "n_tiabs_multi_pmid": int((gold["n_pmids_for_tiab"] > 1).sum()),
    }
    if with_cands:
        prov["n_gold_ids_in_candidates"] = int(gold["gold_in_candidates"].sum())
        prov["candidates_per_tiab"] = shard["n_candidates"].describe().to_dict()
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print(json.dumps(prov, indent=2))
    print(f"[Info] wrote {shard_path} ({len(shard)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
