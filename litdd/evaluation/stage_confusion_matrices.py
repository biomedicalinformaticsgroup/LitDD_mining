"""Per-stage confusion matrices for the pipeline on an annotated fixture.

Each stage is scored as a binary decision over the same abstract population, so the
matrices compose into a funnel:

  stage 1 screen        the abstract is screen-positive; truth = it carries a curated entry
  stage 2 gene gate     a G2P gene is detected in the title and abstract (a candidate exists)
  stage 3 retrieval     the curated entries are among the candidates offered
  stage 4 adjudication  exact set match given the candidates
  end-to-end            all stages composed: exact set match over every fixture abstract

Reads the fixture's ``gold.csv`` (``row_id``, ``true_g2p_ids``, ``n_gold``, ``bert_predict``)
and the adjudication parquets (``row_id``, ``candidates``, ``llm_dis_map``); writes one CSV row
per stage with TP, FP, FN, TN, precision, recall and F1.

    python -m litdd.evaluation.stage_confusion_matrices --gold_csv fixture/gold.csv \\
        --llm_parquet "out/*__llm.parquet" --g2p_csv G2P_DD.csv --out_csv stages.csv
"""
from __future__ import annotations

import argparse
import glob
import math

import pandas as pd

from litdd.evaluation.common import (
    candidate_ids_from_row,
    g2p_ids_regex,
    g2p_ids_split,
    prf,
    x_equivalence_map,
)


def canonical_set(v, canon: dict[str, str]) -> set[str]:
    """Ids in a gold or ``llm_dis_map`` cell, mapped through the X-linked equivalence."""
    ids = g2p_ids_regex(v) or g2p_ids_split(v)
    return {canon.get(i, i) for i in ids}


def cm(tp: int, fp: int, fn: int, tn: int, stage: str, unit: str, note: str = "") -> dict:
    p, r, f = prf(tp, fp, fn, empty=float("nan"))
    return {"stage": stage, "unit": unit, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": None if math.isnan(p) else round(p, 4),
            "recall": None if math.isnan(r) else round(r, 4),
            "f1": None if math.isnan(f) else round(f, 4), "note": note}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold_csv", required=True)
    ap.add_argument("--llm_parquet", required=True, help="path or glob of *__llm.parquet")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--g2p_csv", default=None,
                    help="G2P export for the X-linked equivalence rule; skipped when absent")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    canon = x_equivalence_map(args.g2p_csv) if args.g2p_csv else {}

    gold = pd.read_csv(args.gold_csv)
    gold["row_id"] = gold["row_id"].astype(str)
    paths = sorted(glob.glob(args.llm_parquet)) or [args.llm_parquet]
    llm = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    llm["row_id"] = llm["row_id"].astype(str)
    L = llm.set_index("row_id")
    cands = L["candidates"].apply(lambda c: {canon.get(i, i) for i in candidate_ids_from_row(c)})

    rows = []
    tp = int(((gold.n_gold > 0) & (gold.bert_predict == 1)).sum())
    fp = int(((gold.n_gold == 0) & (gold.bert_predict == 1)).sum())
    fn = int(((gold.n_gold > 0) & (gold.bert_predict == 0)).sum())
    tn = int(((gold.n_gold == 0) & (gold.bert_predict == 0)).sum())
    rows.append(cm(tp, fp, fn, tn, "1 screen (LitDD-BERT)", "abstract",
                   "positive = fires; truth = abstract carries >=1 curated entry"))

    passed = gold[gold.bert_predict == 1]
    keep = passed.row_id.isin(set(llm.row_id))
    tp = int(((passed.n_gold > 0) & keep).sum())
    fp = int(((passed.n_gold == 0) & keep).sum())
    fn = int(((passed.n_gold > 0) & ~keep).sum())
    tn = int(((passed.n_gold == 0) & ~keep).sum())
    rows.append(cm(tp, fp, fn, tn, "2 gene gate (TIAB mention)", "abstract",
                   "on screen-positive abstracts; positive = >=1 G2P gene detected"))

    tp = fn = 0
    for r in passed[passed.n_gold > 0].itertuples():
        if r.row_id not in cands.index:
            continue
        if canonical_set(r.true_g2p_ids, canon) <= cands[r.row_id]:
            tp += 1
        else:
            fn += 1
    rows.append(cm(tp, 0, fn, 0, "3 retrieval (candidates contain the curated entries)",
                   "curated abstract", "TP = every curated entry offered to the LLM"))

    tp = fp = fn = tn = 0
    for r in passed.itertuples():
        if r.row_id not in L.index:
            continue
        g = canonical_set(r.true_g2p_ids, canon)
        p = canonical_set(L.loc[r.row_id, "llm_dis_map"], canon)
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

    tp = fp = fn = tn = 0
    for r in gold.itertuples():
        g = canonical_set(r.true_g2p_ids, canon)
        p = (canonical_set(L.loc[r.row_id, "llm_dis_map"], canon)
             if (r.bert_predict == 1 and r.row_id in L.index) else set())
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
