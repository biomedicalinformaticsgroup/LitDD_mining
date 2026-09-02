#!/usr/bin/env python3
"""Per-stage confusion matrices for the deployed pipeline (manuscript figure/table).

Each stage is scored as a binary decision over the same abstract population, so the
matrices compose into the funnel:

  stage 1 screen        does this abstract describe a curated gene-disease relationship?
                        (positive = screen fires; truth = abstract carries >=1 curated entry)
  stage 2 gene gate     is a G2P gene mentioned in the TIAB? (positive = >=1 candidate)
  stage 3 retrieval     are the curated entries among the candidates offered? (per abstract)
  stage 4 LLM           exact-set adjudication given the candidates
  end-to-end            all stages composed: exact set match over every test abstract

Written for the corrected TIAB gate + no-threshold design.

    python litdd/evaluation/stage_confusion_matrices.py \\
        --run gptoss_test_gatedtb_tnone --fixture revision/llm_eval/annotated_2026 \\
        --out_csv supplementary/stage_confusion_matrices.csv
"""
from __future__ import annotations

import argparse
import glob
import math
import re

import pandas as pd

G2P_ID_RE = re.compile(r"G2P\d+")


def sets(v) -> set[str]:
    """Ids in a gold or llm_dis_map cell; 'NO MATCH'/None/'' are the empty set."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return set()
    txt = str(v).strip()
    if not txt or txt.upper() == "NO MATCH" or txt == "nan":
        return set()
    return set(G2P_ID_RE.findall(txt)) or {s for s in txt.split(";") if s and s != "nan"}


def cm(tp, fp, fn, tn, stage, unit, note=""):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p and r and (p + r) else float("nan")
    return {"stage": stage, "unit": unit, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": None if math.isnan(p) else round(p, 4),
            "recall": None if math.isnan(r) else round(r, 4),
            "f1": None if math.isnan(f) else round(f, 4), "note": note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    gold = pd.read_csv(f"{args.fixture}/gold.csv")
    gold["row_id"] = gold["row_id"].astype(str)
    llm = pd.read_parquet(glob.glob(f"revision/llm_eval/runs/{args.run}/*__llm.parquet")[0])
    llm["row_id"] = llm["row_id"].astype(str)
    L = llm.set_index("row_id")
    cands = L["top5_cross"].apply(lambda c: {G2P_ID_RE.search(i["label"]).group(0) for i in list(c)})

    rows = []
    # stage 1: screen
    tp = int(((gold.n_gold > 0) & (gold.bert_predict == 1)).sum())
    fp = int(((gold.n_gold == 0) & (gold.bert_predict == 1)).sum())
    fn = int(((gold.n_gold > 0) & (gold.bert_predict == 0)).sum())
    tn = int(((gold.n_gold == 0) & (gold.bert_predict == 0)).sum())
    rows.append(cm(tp, fp, fn, tn, "1 screen (LitDD-BERT)", "abstract",
                   "positive = fires; truth = abstract carries >=1 curated entry"))
    # stage 2: gene gate, over abstracts the screen passed
    passed = gold[gold.bert_predict == 1]
    keep = passed.row_id.isin(set(llm.row_id))
    tp = int(((passed.n_gold > 0) & keep).sum())
    fp = int(((passed.n_gold == 0) & keep).sum())
    fn = int(((passed.n_gold > 0) & ~keep).sum())
    tn = int(((passed.n_gold == 0) & ~keep).sum())
    rows.append(cm(tp, fp, fn, tn, "2 gene gate (TIAB mention)", "abstract",
                   "on screen-positive abstracts; positive = >=1 G2P gene detected"))
    # stage 3: retrieval (are the curated entries offered?)
    tp = fn = 0
    for r in passed[passed.n_gold > 0].itertuples():
        if r.row_id not in cands.index:
            continue
        if sets(r.true_g2p_ids) <= cands[r.row_id]:
            tp += 1
        else:
            fn += 1
    rows.append(cm(tp, 0, fn, 0, "3 retrieval (candidates contain the curated entries)",
                   "curated abstract", "TP = every curated entry offered to the LLM"))
    # stage 4: LLM adjudication on what it was given
    tp = fp = fn = tn = 0
    for r in passed.itertuples():
        if r.row_id not in L.index:
            continue
        g = sets(r.true_g2p_ids)
        p = sets(L.loc[r.row_id, "llm_dis_map"])
        if g and p == g:
            tp += 1
        elif g and not p:
            fn += 1
        elif g:
            fp += 1
            fn += 1
        elif p:
            fp += 1
        else:
            tn += 1
    rows.append(cm(tp, fp, fn, tn, "4 LLM adjudication (exact set)", "abstract",
                   "on abstracts reaching the LLM; a wrong set counts as both FP and FN"))
    # end to end
    tp = fp = fn = tn = 0
    for r in gold.itertuples():
        g = sets(r.true_g2p_ids)
        p = sets(L.loc[r.row_id, "llm_dis_map"]) if (r.bert_predict == 1 and r.row_id in L.index) else set()
        if g and p == g:
            tp += 1
        elif g and not p:
            fn += 1
        elif g:
            fp += 1
            fn += 1
        elif p:
            fp += 1
        else:
            tn += 1
    rows.append(cm(tp, fp, fn, tn, "END-TO-END (all stages)", "abstract",
                   "exact set match over every test abstract"))
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
