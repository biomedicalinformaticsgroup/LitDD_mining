#!/usr/bin/env python3
"""Score an LLM-adjudication run against the annotated set (multi-disease aware).

Input: one or more ``*__llm.parquet`` written by ``litdd/pipeline/llm_map.py`` over the
fixture from ``build_llm_eval_shards.py``, plus that fixture's ``gold.csv`` (and optionally
``pairs.csv``). A TIAB may carry several gold G2P ids and the LLM may return several
(``G2Pa;G2Pb``); every view below treats the answer as a SET.

Metric views (all reported side by side so each number is traceable to a prior figure):

  paper_legacy   the original manuscript's scorer (train_test/test_metrics.py) reproduced
                 exactly: per-TIAB exact set match, end-to-end with the screen and the
                 cross-encoder > cutoff gate, GeneReviews excluded. NOTE its quirk: a
                 positive TIAB that also had labelled negative pairs is scored twice, once
                 as a positive row and once as a phantom negative row (on which any mapping
                 is an FP). Kept only for continuity with the published number.
  tiab_exact     per-TIAB exact set match, LLM stage alone (given the candidates), no gates.
  id_micro       per-(TIAB, G2P id) micro P/R/F1: TP=|pred ∩ gold|, FP=|pred − gold|,
                 FN=|gold − pred|. The natural multi-disease metric; Wilson 95% CIs.
  id_micro_gated as id_micro after the deployed cross-encoder gate (predicted id's score
                 ≥ cutoff) -- what final_data_clean.py would keep, minus the gene check.
  id_micro_screen_gated
                 additionally requires bert_predict == 1 (the LLM never sees screen
                 negatives in deployment).
  pair_level     Fabian's harness definition (eval_basic_output / evaluate_v2): over labelled
                 (TIAB, id) pairs that the retriever offered, pred=1 iff the id is in the
                 answer set. The positives the retriever did NOT offer are counted, not
                 silently dropped.

Plus answer-quality rates (NO MATCH, unparsed, format-invalid, hallucinated, UNCERTAIN,
truncated), token/throughput figures from run_meta.json, and strata: single- vs
multi-gold TIABs, and TIABs whose candidates contain ≥2 entries of the same gene (the only
slice where disambiguation is non-trivial; R2-D2).

Outputs: <out_prefix>_summary.json, <out_prefix>_per_tiab.csv, and a printed table.

    python litdd/evaluation/llm_adjudication_eval.py \\
        --llm_parquet "revision/llm_eval/runs/gptoss_vanilla/*__llm.parquet" \\
        --gold_csv revision/llm_eval/annotated_2025/gold.csv \\
        --pairs_csv revision/llm_eval/annotated_2025/pairs.csv \\
        --out_prefix revision/llm_eval/runs/gptoss_vanilla/eval --label gptoss_vanilla

Paired comparison of two runs (exact McNemar on per-TIAB correctness, paired bootstrap on
the id-micro F1 difference):

    python litdd/evaluation/llm_adjudication_eval.py --compare A_per_tiab.csv B_per_tiab.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from litdd.evaluation.compare_models import mcnemar_exact  # noqa: E402

G2P_ID_RE = re.compile(r"G2P\d+")
NO_MATCH = "NO MATCH"


# ----------------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
           "f1": round(f, 4)}
    out["precision_ci95"] = [round(x, 4) for x in wilson(tp, tp + fp)]
    out["recall_ci95"] = [round(x, 4) for x in wilson(tp, tp + fn)]
    return out


def parse_set(v) -> set[str]:
    """``llm_dis_map`` -> set of ids ('NO MATCH', None, '' -> empty set)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return set()
    s = str(v).strip()
    if not s or s.upper() == NO_MATCH:
        return set()
    return set(G2P_ID_RE.findall(s))


def gold_set(v) -> set[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return set()
    return {x for x in str(v).split(";") if x}


def _as_list(top5) -> list:
    """A parquet list<struct> cell arrives as a numpy array; None/NaN -> []."""
    if top5 is None or (isinstance(top5, float) and math.isnan(top5)):
        return []
    return list(top5)


def candidate_scores(top5) -> dict[str, float]:
    out = {}
    for item in _as_list(top5):
        if isinstance(item, dict):
            lab, sc = item.get("label"), item.get("score")
        else:
            lab, sc = item[0], item[1]
        m = G2P_ID_RE.search(str(lab) or "")
        if m:
            out[m.group(0)] = float(sc)
    return out


def candidate_genes(top5) -> list[str]:
    genes = []
    for item in _as_list(top5):
        lab = item.get("label") if isinstance(item, dict) else item[0]
        parts = str(lab).split(" - ")
        genes.append(parts[1].strip() if len(parts) > 1 else "")
    return genes


# ----------------------------------------------------------------------------- scoring
def per_tiab_table(llm: pd.DataFrame, gold: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    key = "row_id" if "row_id" in llm.columns and "row_id" in gold.columns else "pmid"
    llm = llm.copy()
    llm[key] = llm[key].astype(str)
    gold = gold.copy()
    gold[key] = gold[key].astype(str)
    df = gold.merge(llm, on=key, how="left", suffixes=("", "_llm"), validate="one_to_one")
    missing = df["generated_text"].isna().sum() if "generated_text" in df.columns else len(df)
    if missing:
        print(f"[WARN] {missing} gold TIABs have no LLM row (not generated / join failure)")

    rows = []
    for r in df.itertuples(index=False):
        g = gold_set(r.true_g2p_ids)
        p = parse_set(getattr(r, "llm_dis_map", None))
        scores = candidate_scores(getattr(r, "top5_cross", None))
        genes = candidate_genes(getattr(r, "top5_cross", None))
        p_gated = {i for i in p if scores.get(i, 0.0) >= cutoff}
        bert = int(getattr(r, "bert_predict", 1) or 0)
        p_screen = p_gated if bert == 1 else set()
        raw = getattr(r, "llm_dis_map", None)
        has_row = isinstance(getattr(r, "generated_text", None), str)
        rows.append({
            key: getattr(r, key), "pmid": r.pmid,
            "n_gold": len(g), "gold": ";".join(sorted(g)), "pred": ";".join(sorted(p)),
            "cand_ids": ";".join(sorted(scores)),  # every candidate the run scored (pre-gate)
            "multi_gold": len(g) > 1,
            "cands_share_gene": len(genes) != len(set(genes)),
            "n_candidates": len(scores),
            "bert_predict": bert, "max_cross": max(scores.values(), default=float("nan")),
            "genereviews": bool(getattr(r, "genereviews", False)),
            "exact_correct": p == g,
            "id_tp": len(p & g), "id_fp": len(p - g), "id_fn": len(g - p),
            "id_tp_gated": len(p_gated & g), "id_fp_gated": len(p_gated - g),
            "id_fn_gated": len(g - p_gated),
            "id_tp_screen": len(p_screen & g), "id_fp_screen": len(p_screen - g),
            "id_fn_screen": len(g - p_screen),
            "no_match": (str(raw).upper() == NO_MATCH) if raw is not None
            and not (isinstance(raw, float) and math.isnan(raw)) else False,
            # answer-quality flags apply only to rows the LLM stage actually produced;
            # rows dropped upstream are counted once, under no_llm_row.
            "unparsed": has_row and (raw is None or (isinstance(raw, float) and math.isnan(raw))),
            "format_invalid": has_row and getattr(r, "answer_format_valid", None) is False,
            "uncertain": has_row and getattr(r, "answer_uncertain", None) is True,
            "hallucinated": getattr(r, "answer_ids_in_candidates", None) is False,
            "truncated": getattr(r, "finish_reason", None) == "length",
            # Reached the LLM stage but no candidate passed the pre-LLM score gate: recorded as
            # NO MATCH without a model call (gene gate -> cross-encoder -> score gate -> LLM).
            "skipped_no_candidates": getattr(r, "finish_reason", None) == "skipped",
            # No LLM row at all: the TIAB never reached the LLM (dropped by an upstream hard
            # gate such as the gene filter). Its gold ids count as FN in every view.
            "no_llm_row": not has_row,
            "gen_tokens": getattr(r, "gen_tokens", None),
            "prompt_tokens": getattr(r, "prompt_tokens", None),
        })
    return pd.DataFrame(rows)


def view_id_micro(t: pd.DataFrame, suffix: str = "") -> dict:
    return prf(int(t[f"id_tp{suffix}"].sum()), int(t[f"id_fp{suffix}"].sum()),
               int(t[f"id_fn{suffix}"].sum()))


def view_tiab_exact(t: pd.DataFrame) -> dict:
    has_gold = t["n_gold"] > 0
    has_pred = t["pred"] != ""
    tp = int((has_gold & t["exact_correct"]).sum())
    fp = int((has_pred & ~t["exact_correct"]).sum())
    fn = int((has_gold & ~t["exact_correct"]).sum())
    tn = int((~has_gold & ~has_pred).sum())
    out = prf(tp, fp, fn)
    out["tn"] = tn
    out["accuracy"] = round(float(t["exact_correct"].mean()), 4) if len(t) else float("nan")
    return out


def view_paper_legacy(t: pd.DataFrame, pairs: pd.DataFrame | None, cutoff: float) -> dict:
    """train_test/test_metrics.py, reproduced row for row (see module docstring)."""
    key = "row_id" if "row_id" in t.columns else "pmid"
    has_neg_pair = set()
    if pairs is not None:
        has_neg_pair = set(pairs.loc[pairs["label"] == 0, key].astype(str))
    tp = tn = fp = fn = 0
    for r in t[~t["genereviews"]].itertuples(index=False):
        rid = str(getattr(r, key))
        cross = 1 if (not math.isnan(r.max_cross) and r.max_cross > cutoff) else 0
        pred_bert = r.bert_predict
        llm_set = set(r.pred.split(";")) - {""}
        no_match = r.no_match
        legacy_rows = []
        if r.n_gold > 0:
            legacy_rows.append((1, set(r.gold.split(";")) - {""}))
            if rid in has_neg_pair:
                legacy_rows.append((0, set()))
        else:
            legacy_rows.append((0, set()))
        for label, true_set in legacy_rows:
            if label == 1 and pred_bert == 1 and cross == 1 and llm_set == true_set:
                tp += 1
            elif (label == 0 and pred_bert == 0) or (label == 0 and pred_bert == 1 and no_match) \
                    or (label == 0 and pred_bert == 1 and cross == 0):
                tn += 1
            elif label == 1:
                fn += 1
            else:
                fp += 1
    out = prf(tp, fp, fn)
    out["tn"] = tn
    out["note"] = ("exact reproduction of test_metrics.py incl. phantom negative rows; "
                   "not comparable to the LLM-only views")
    return out


def view_pair_level(t: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    key = "row_id" if "row_id" in t.columns else "pmid"
    pred_by = {str(getattr(r, key)): set(r.pred.split(";")) - {""} for r in t.itertuples(index=False)}
    pairs = pairs.copy()
    pairs[key] = pairs[key].astype(str)
    # "Offered" = the retriever put the entry in front of the pipeline. Fixtures built with
    # --candidates none carry no in_candidates column, so derive it from the candidates the
    # run actually scored (top5_cross of the LLM parquet, before any pre-LLM score gate).
    if "in_candidates" not in pairs.columns or pairs["in_candidates"].isna().all():
        cands_by = {str(getattr(r, key)): set(str(r.cand_ids).split(";")) - {"", "nan"}
                    for r in t.itertuples(index=False)}
        pairs["in_candidates"] = [g in cands_by.get(k, set())
                                  for k, g in zip(pairs[key], pairs["g2p_id"])]
    pairs["in_candidates"] = pairs["in_candidates"].fillna(False).astype(bool)
    offered = pairs[pairs["in_candidates"]]
    tp = fp = fn = tn = 0
    for r in offered.itertuples(index=False):
        pred = 1 if r.g2p_id in pred_by.get(str(getattr(r, key)), set()) else 0
        if r.label == 1 and pred == 1:
            tp += 1
        elif r.label == 0 and pred == 1:
            fp += 1
        elif r.label == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    out = prf(tp, fp, fn)
    out["tn"] = tn
    out["labelled_pairs"] = int(len(pairs))
    out["pairs_offered_by_retriever"] = int(len(offered))
    out["positive_pairs_not_offered"] = int(((pairs["label"] == 1) & ~pairs["in_candidates"]).sum())
    out["note"] = "Fabian harness definition (evaluate_v2) over offered pairs"
    return out


def rates(t: pd.DataFrame) -> dict:
    n = len(t)
    out = {"n_tiabs": n}
    for c in ("no_match", "unparsed", "format_invalid", "uncertain", "hallucinated", "truncated",
              "no_llm_row", "skipped_no_candidates"):
        out[f"{c}_rate"] = round(float(t[c].mean()), 4) if n else float("nan")
        out[f"{c}_n"] = int(t[c].sum())
    for c in ("gen_tokens", "prompt_tokens"):
        s = pd.to_numeric(t[c], errors="coerce").dropna()
        if len(s):
            out[f"{c}_mean"] = round(float(s.mean()), 1)
            out[f"{c}_p95"] = round(float(s.quantile(0.95)), 1)
            out[f"{c}_max"] = int(s.max())
    return out


def strata(t: pd.DataFrame) -> dict:
    out = {}
    for name, mask in (("single_gold", (t["n_gold"] == 1)), ("multi_gold", t["multi_gold"]),
                       ("no_gold", t["n_gold"] == 0),
                       ("cands_share_gene", t["cands_share_gene"]),
                       ("cands_distinct_genes", ~t["cands_share_gene"])):
        sub = t[mask]
        out[name] = {"n_tiabs": int(len(sub)),
                     "exact_accuracy": round(float(sub["exact_correct"].mean()), 4) if len(sub) else None,
                     "id_micro": view_id_micro(sub) if len(sub) else None}
    return out


def load_run_meta(paths: list[str]) -> dict:
    metas = []
    for p in paths:
        mp = p.replace("__llm.parquet", "__llm.run_meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                metas.append(json.load(f))
    if not metas:
        return {}
    m = metas[0]
    keep = {k: m.get(k) for k in ("model", "reasoning_effort", "use_chat_template", "threads",
                                  "min_score", "max_candidates",
                                  "temperature", "top_p", "max_tokens", "max_model_len", "seed",
                                  "dtype", "git_sha", "image", "versions", "gpu", "source")}
    keep["rows_per_s"] = m.get("rows_per_s")
    keep["gen_tokens_per_s"] = m.get("gen_tokens_per_s")
    keep["generation_s_total"] = round(sum(x.get("generation_s") or 0 for x in metas), 1)
    keep["shards"] = len(metas)
    return keep


# ----------------------------------------------------------------------------- compare
def compare(a_csv: str, b_csv: str, n_boot: int = 2000, seed: int = 0) -> dict:
    a = pd.read_csv(a_csv)
    b = pd.read_csv(b_csv)
    key = "row_id" if "row_id" in a.columns else "pmid"
    m = a.merge(b, on=key, suffixes=("_a", "_b"), validate="one_to_one")
    labels = [1] * len(m)
    ca = m["exact_correct_a"].astype(int).tolist()
    cb = m["exact_correct_b"].astype(int).tolist()
    b_only, c_only, p = mcnemar_exact(labels, ca, cb)

    def f1_of(idx, s):
        tp = m[f"id_tp{s}"].iloc[idx].sum()
        fp = m[f"id_fp{s}"].iloc[idx].sum()
        fn = m[f"id_fn{s}"].iloc[idx].sum()
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    rng = random.Random(seed)
    n = len(m)
    all_idx = list(range(n))
    diffs = sorted(f1_of(idx := [rng.randrange(n) for _ in range(n)], "_a") - f1_of(idx, "_b")
                   for _ in range(n_boot))
    return {
        "a": a_csv, "b": b_csv, "n_paired_tiabs": n,
        "exact_accuracy_a": round(float(m["exact_correct_a"].mean()), 4),
        "exact_accuracy_b": round(float(m["exact_correct_b"].mean()), 4),
        "mcnemar": {"a_right_b_wrong": b_only, "a_wrong_b_right": c_only, "p_exact": p},
        "id_micro_f1_a": round(f1_of(all_idx, "_a"), 4),
        "id_micro_f1_b": round(f1_of(all_idx, "_b"), 4),
        "id_micro_f1_diff_a_minus_b": round(f1_of(all_idx, "_a") - f1_of(all_idx, "_b"), 4),
        "id_micro_f1_diff_ci95": [round(diffs[int(0.025 * n_boot)], 4),
                                  round(diffs[int(0.975 * n_boot) - 1], 4)],
    }


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm_parquet", help="path or glob of *__llm.parquet")
    ap.add_argument("--gold_csv")
    ap.add_argument("--pairs_csv", default=None)
    ap.add_argument("--score_cutoff", type=float, default=0.9)
    ap.add_argument("--out_prefix")
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A_PER_TIAB", "B_PER_TIAB"))
    args = ap.parse_args()

    if args.compare:
        res = compare(*args.compare)
        print(json.dumps(res, indent=2))
        if args.out_prefix:
            with open(f"{args.out_prefix}_compare.json", "w") as f:
                json.dump(res, f, indent=2)
        return 0

    if not (args.llm_parquet and args.gold_csv and args.out_prefix):
        ap.error("--llm_parquet, --gold_csv and --out_prefix are required")
    paths = sorted(glob.glob(args.llm_parquet)) or [args.llm_parquet]
    llm = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    gold = pd.read_csv(args.gold_csv)
    pairs = pd.read_csv(args.pairs_csv) if args.pairs_csv else None

    t = per_tiab_table(llm, gold, args.score_cutoff)
    summary = {
        "label": args.label or os.path.basename(args.out_prefix),
        "llm_parquet": paths, "gold_csv": args.gold_csv, "score_cutoff": args.score_cutoff,
        "run": load_run_meta(paths),
        "rates": rates(t),
        "paper_legacy": view_paper_legacy(t, pairs, args.score_cutoff),
        "tiab_exact": view_tiab_exact(t),
        "id_micro": view_id_micro(t),
        "id_micro_gated": view_id_micro(t, "_gated"),
        "id_micro_screen_gated": view_id_micro(t, "_screen"),
        # The LLM stage alone on the rows it actually receives in deployment (screen
        # positives), with no cross-encoder gate: the cleanest vanilla-vs-contextualised
        # comparison, since the 2,000+ screen-negative TIABs never reach the LLM.
        "id_micro_screen_positives_only": view_id_micro(t[t["bert_predict"] == 1]),
        "tiab_exact_screen_positives_only": view_tiab_exact(t[t["bert_predict"] == 1]),
        "pair_level": view_pair_level(t, pairs) if pairs is not None else None,
        "strata": strata(t),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    t.to_csv(f"{args.out_prefix}_per_tiab.csv", index=False)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n== {summary['label']}  ({len(t)} TIABs; model {summary['run'].get('model')}; "
          f"threads {summary['run'].get('threads')}; effort {summary['run'].get('reasoning_effort')})")
    print(f"{'view':<24}{'P':>8}{'R':>8}{'F1':>8}   tp/fp/fn")
    for name in ("paper_legacy", "tiab_exact", "id_micro", "id_micro_gated",
                 "id_micro_screen_gated", "id_micro_screen_positives_only",
                 "tiab_exact_screen_positives_only", "pair_level"):
        v = summary[name]
        if v:
            print(f"{name:<24}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}   "
                  f"{v['tp']}/{v['fp']}/{v['fn']}")
    r = summary["rates"]
    print(f"rates: no_match {r['no_match_rate']:.3f}  unparsed {r['unparsed_rate']:.3f}  "
          f"format_invalid {r['format_invalid_rate']:.3f}  hallucinated {r['hallucinated_rate']:.3f}  "
          f"uncertain {r['uncertain_rate']:.3f}  truncated {r['truncated_rate']:.3f}  "
          f"no_llm_row {r['no_llm_row_rate']:.3f}  skipped_by_gate {r['skipped_no_candidates_rate']:.3f}  "
          f"gen_tokens mean/p95 {r.get('gen_tokens_mean')}/{r.get('gen_tokens_p95')}")
    for k, v in summary["strata"].items():
        if v["id_micro"]:
            print(f"  stratum {k:<22} n={v['n_tiabs']:<5} exact_acc {v['exact_accuracy']:.4f}  "
                  f"id_micro F1 {v['id_micro']['f1']:.4f}")
    print(f"wrote {args.out_prefix}_summary.json / _per_tiab.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
