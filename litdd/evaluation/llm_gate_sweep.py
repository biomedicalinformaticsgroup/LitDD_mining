#!/usr/bin/env python3
"""Pre-LLM cross-encoder threshold sweep: end-of-cascade metrics per gate value.

Each run is the deployment cascade (gene gate -> cross-encoder on every entry of the
detected genes -> candidates scoring >= t -> LLM) with a different t, all on the same
shards. This scores every run with llm_adjudication_eval.py and tabulates, per t:

  llm_rows          abstracts that reached the LLM (the rest had no candidate >= t)
  gold_ids_shown    curated entries that were shown to the LLM at all (retriever + gate)
  screen-positive per-(abstract, entry) P / R / F1 (what the cascade returns)
  all-TIAB precision (LLM-alone rejection of gate-passing near-miss negatives)
  false entries, NO MATCH rate, mean generated tokens, rows/s

    python litdd/evaluation/llm_gate_sweep.py \\
        --runs "revision/llm_eval/runs/gptoss_2026_gated*" \\
        --gold_csv revision/llm_eval/annotated_2026/gold.csv \\
        --pairs_csv revision/llm_eval/annotated_2026/pairs.csv \\
        --out_dir revision/llm_eval/eval_2026 --out_csv revision/llm_eval/eval_2026/gate_sweep.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
G2P_ID_RE = re.compile(r"G2P\d+")


def gold_ids_shown(llm_parquet: str, gold: pd.DataFrame) -> tuple[int, int]:
    """(curated entries shown to the LLM, curated entries scored by the cross-encoder)."""
    df = pd.read_parquet(llm_parquet, columns=["row_id", "topk_cross_lgmde", "top5_cross"])
    df["row_id"] = df["row_id"].astype(str)
    shown = {r.row_id: {G2P_ID_RE.search(str(x)).group(0) for x in list(r.topk_cross_lgmde)
                        if G2P_ID_RE.search(str(x))} for r in df.itertuples(index=False)}
    scored = {r.row_id: {G2P_ID_RE.search(str(i["label"])).group(0) for i in list(r.top5_cross)}
              for r in df.itertuples(index=False)}
    n_shown = n_scored = 0
    for r in gold[gold["n_gold"] > 0].itertuples(index=False):
        for g in str(r.true_g2p_ids).split(";"):
            if not g:
                continue
            n_shown += g in shown.get(str(r.row_id), set())
            n_scored += g in scored.get(str(r.row_id), set())
    return n_shown, n_scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="glob of run directories")
    ap.add_argument("--gold_csv", required=True)
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_dir", required=True, help="where per-run eval_* files go (user-writable)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    gold = pd.read_csv(args.gold_csv)
    gold["row_id"] = gold["row_id"].astype(str)
    n_gold_ids = int(gold["n_gold"].sum())
    rows = []
    for run_dir in sorted(glob.glob(args.runs)):
        run = os.path.basename(run_dir)
        parquets = glob.glob(os.path.join(run_dir, "*__llm.parquet"))
        if not parquets:
            print(f"[skip] {run}: no LLM parquet yet")
            continue
        out_prefix = os.path.join(args.out_dir, run, "eval")
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        subprocess.run([sys.executable, os.path.join(ROOT, "litdd", "evaluation", "llm_adjudication_eval.py"),
                        "--llm_parquet", os.path.join(run_dir, "*__llm.parquet"),
                        "--gold_csv", args.gold_csv, "--pairs_csv", args.pairs_csv,
                        "--out_prefix", out_prefix, "--label", run],
                       check=True, capture_output=True, text=True)
        with open(out_prefix + "_summary.json") as f:
            s = json.load(f)
        with open(parquets[0].replace("__llm.parquet", "__llm.run_meta.json")) as f:
            m = json.load(f)
        shown, scored = gold_ids_shown(parquets[0], gold)
        sp = s["id_micro_screen_positives_only"]
        allt = s["id_micro"]
        rows.append({
            "run": run, "min_score": m.get("min_score"),
            "rows_total": m.get("rows_total"),
            "llm_rows": m["rows_total"] - m.get("rows_skipped_no_candidates", 0),
            "candidates_removed": m.get("candidates_removed_by_min_score"),
            "gold_ids_scored_by_ce": scored, "gold_ids_shown_to_llm": shown,
            "gold_ids_total": n_gold_ids,
            "screenpos_precision": sp["precision"], "screenpos_recall": sp["recall"],
            "screenpos_f1": sp["f1"], "screenpos_tp": sp["tp"], "screenpos_fp": sp["fp"],
            "screenpos_fn": sp["fn"],
            "all_tiab_precision": allt["precision"], "all_tiab_recall": allt["recall"],
            "all_tiab_fp": allt["fp"],
            "no_match_rate": s["rates"]["no_match_rate"],
            "gen_tokens_mean": m.get("gen_tokens_mean"), "rows_per_s": m.get("rows_per_s"),
            "generation_s": m.get("generation_s"),
        })
    df = pd.DataFrame(rows).sort_values("min_score", na_position="first")
    df.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 250)
    print(df[["min_score", "llm_rows", "gold_ids_shown_to_llm", "screenpos_precision",
              "screenpos_recall", "screenpos_f1", "screenpos_fp", "all_tiab_precision",
              "all_tiab_fp", "no_match_rate", "rows_per_s"]].to_string(index=False))
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
