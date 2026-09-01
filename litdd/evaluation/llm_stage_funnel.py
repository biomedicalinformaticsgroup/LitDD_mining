#!/usr/bin/env python3
"""Stage-by-stage funnel of the adjudication cascade on the annotated test split.

For each pipeline the abstracts flow through its stages in deployment order; after every
stage we report the cumulative exact-match precision / recall / F1 and how many curated
abstracts (potential true positives) that stage removed.

  alive curated abstract  = its curated entries can still be returned exactly
                            (before the LLM: screen passed, curated entries still among the
                             candidates; from the LLM on: the answer set equals the curated set
                             and every one of its entries survives the later gates)
  alive non-curated / wrong = the abstract would still emit a mapping after this stage
  TP = alive curated · FN = curated − TP · FP = alive non-curated + curated alive-but-wrong

Original order:  screen -> cross-encoder full-panel top-5 -> LLM -> gene-mention check
                 (PubTator symbol match, as final_data_clean.py) -> score >= 0.9
Revised order:   screen -> gene gate (PubTator + HGNC names) -> cross-encoder on every entry
                 of the detected genes -> score >= t -> LLM

    python litdd/evaluation/llm_stage_funnel.py --out_csv revision/llm_eval/eval_2026/stage_funnel.csv
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from litdd.threads import load_g2p  # noqa: E402

REF = ("/home/eidf128/eidf128/shared/export/michael/ddg2p_pubmed2diseasemodel_CLEAN/"
       "clean_pipeline/train_test")
G2P_ID_RE = re.compile(r"G2P\d+")


def sets(v) -> set[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return set()
    return {x for x in str(v).split(";") if x and x != "nan"}


def scores(cell) -> dict[str, float]:
    return {G2P_ID_RE.search(str(i["label"])).group(0): float(i["score"]) for i in list(cell)}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def id2gene(g2p_csv):
    d = load_g2p(g2p_csv).drop_duplicates("g2p id")
    return dict(zip(d["g2p id"].astype(str), d["gene symbol"].astype(str)))


def funnel(name, gold, stages):
    """stages: list of (stage_name, fn(row_id, state) -> state or None). state carries
    'pred' (set or None before the LLM) and is None once the abstract is dead."""
    rows = []
    n_cur = int((gold["n_gold"] > 0).sum())
    state = {r.row_id: {"gold": sets(r.true_g2p_ids), "pred": None, "gold_alive": True}
             for r in gold.itertuples()}
    prev_tp = n_cur
    for sname, fn in stages:
        for rid, st in state.items():
            if st is None:
                continue
            state[rid] = fn(rid, st)
        tp = fp = 0
        for rid, st in state.items():
            if st is None:
                continue
            if st["gold"]:
                if st["pred"] is None:
                    tp += st["gold_alive"]
                    fp += not st["gold_alive"] and False  # unmapped-yet: not an FP before the LLM
                else:
                    ok = st["pred"] == st["gold"]
                    tp += ok
                    fp += (not ok) and len(st["pred"]) > 0
            else:
                fp += 1 if (st["pred"] is None or len(st["pred"]) > 0) else 0
        fn_ = n_cur - tp
        p, r, f = prf(tp, fp, fn_)
        rows.append({"pipeline": name, "stage": sname, "TP_alive": tp, "FP_alive": fp, "FN": fn_,
                     "precision": p, "recall": r, "f1": f, "TP_lost_at_stage": prev_tp - tp,
                     "abstracts_alive": sum(1 for s in state.values() if s is not None
                                            and (s["pred"] is None or len(s["pred"]) > 0))})
        prev_tp = tp
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    all_rows = []

    # ------------------------------------------------------------------ original pipeline
    g25 = id2gene(f"{REF}/G2P_DD_2025-02-15.csv")
    gold = pd.read_csv("revision/llm_eval/annotated_2025_deployedscreen/gold.csv")
    gold["row_id"] = gold["row_id"].astype(str)
    per = pd.read_csv("revision/llm_eval/eval_2025_deployedscreen/deepseek_deployed/eval_per_tiab.csv")
    per["row_id"] = per["row_id"].astype(str)
    pred25 = dict(zip(per["row_id"], per["pred"].map(sets)))
    bert25 = dict(zip(gold["row_id"], gold["bert_predict"]))
    fix = pd.read_parquet("revision/llm_eval/annotated_2025/shards/annotated_test.parquet")
    fix["row_id"] = fix["row_id"].astype(str)
    top5 = dict(zip(fix["row_id"], fix["top5_cross"].map(scores)))
    cand25 = pd.read_parquet("revision/llm_eval/annotated_2025_genegated/candidates.parquet")
    cand25["row_id"] = cand25["row_id"].astype(str)
    # PubTator symbol matches only = the original final_data_clean gene check
    pub_genes = {}
    for r in cand25.itertuples():
        ids = list(r.candidate_g2p_ids)
        src = list(r.candidate_sources)
        pub_genes[r.row_id] = {g25.get(i) for i, s in zip(ids, src) if s == "symbol_match"}

    def o_screen(rid, st):
        return st if bert25.get(rid, 0) == 1 else None

    def o_top5(rid, st):
        st["gold_alive"] = st["gold"] <= set(top5.get(rid, {}))
        return st

    def o_llm(rid, st):
        st["pred"] = pred25.get(rid, set())
        return st

    def o_genecheck(rid, st):
        st["pred"] = {i for i in st["pred"] if g25.get(i) in pub_genes.get(rid, set())}
        return st

    def o_gate(rid, st):
        sc = top5.get(rid, {})
        st["pred"] = {i for i in st["pred"] if sc.get(i, 0.0) >= 0.9}
        return st

    all_rows += funnel("original (verified screen)", gold, [
        ("1 screen (BERT positive)", o_screen),
        ("2 cross-encoder top-5 contains curated entries", o_top5),
        ("3 LLM exact set", o_llm),
        ("4 gene-mention check (PubTator)", o_genecheck),
        ("5 score >= 0.9", o_gate),
    ])

    # ------------------------------------------------------------------ revised pipeline
    gold26 = pd.read_csv("revision/llm_eval/annotated_2026/gold.csv")
    gold26["row_id"] = gold26["row_id"].astype(str)
    bert26 = dict(zip(gold26["row_id"], gold26["bert_predict"]))
    cand26 = pd.read_parquet("revision/llm_eval/annotated_2026/candidates_gated.parquet")
    cand26["row_id"] = cand26["row_id"].astype(str)
    gate_entries = {r.row_id: set(list(r.candidate_g2p_ids)) for r in cand26.itertuples()}
    for label, run_dir, eval_dir, thr in [
        ("revised, no gate", "gptoss_2026_gated_tnone", "gptoss_2026_gated_tnone", None),
        ("revised, gate 0.5", "gptoss_2026_gated_t05", "gptoss_2026_gated_t05", 0.5),
        ("revised, gate 0.9 (deployed)", "gptoss_2026_gated", "gptoss_2026_gated", 0.9),
    ]:
        llm = pd.read_parquet(f"revision/llm_eval/runs/{run_dir}/candidates_gated_crossencoded_shard0-of-1__llm.parquet")
        llm["row_id"] = llm["row_id"].astype(str)
        sc_all = dict(zip(llm["row_id"], llm["top5_cross"].map(scores)))
        per = pd.read_csv(f"revision/llm_eval/eval_2026/{eval_dir}/eval_per_tiab.csv")
        per["row_id"] = per["row_id"].astype(str)
        pred = dict(zip(per["row_id"], per["pred"].map(sets)))

        def n_screen(rid, st):
            return st if bert26.get(rid, 0) == 1 else None

        def n_genegate(rid, st):
            if rid not in gate_entries:
                return None
            st["gold_alive"] = st["gold"] <= gate_entries[rid]
            return st

        def n_cegate(rid, st, thr=thr):
            sc = sc_all.get(rid, {})
            kept = {i for i, s in sc.items() if thr is None or s >= thr}
            if not kept:
                return None
            st["gold_alive"] = st["gold_alive"] and st["gold"] <= kept
            return st

        def n_llm(rid, st):
            st["pred"] = pred.get(rid, set())
            return st

        all_rows += funnel(label, gold26, [
            ("1 screen (BERT positive)", n_screen),
            ("2 gene gate (PubTator + HGNC): abstract kept & curated entries among its genes' entries", n_genegate),
            (f"3 cross-encoder score gate (>= {thr if thr is not None else 'none'})", n_cegate),
            ("4 LLM exact set", n_llm),
        ])

    df = pd.DataFrame(all_rows)
    df.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
