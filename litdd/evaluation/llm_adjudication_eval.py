"""Score an adjudication run against the annotated fixture.

Reads the ``*__llm.parquet`` files written by ``litdd.pipeline.llm_map`` over a fixture from
``build_llm_eval_shards.py``, the fixture's ``gold.csv`` and optionally its ``pairs.csv``.
An abstract may carry several curated G2P ids and the adjudicator may return several
(``G2Pa;G2Pb``); every view treats the answer as a set.

Metric views:

  end_to_end_exact  screen positive and exact set match, over every fixture abstract
  tiab_exact        exact set match given the candidates, adjudication stage alone
  id_micro          per-(abstract, id) micro precision, recall and F1 with Wilson intervals
  id_micro_screen   id_micro restricted to ids of screen-positive abstracts
  pair_level        per labelled (abstract, id) pair offered by the gate: predicted iff the id
                    is in the answer set; positive pairs the gate did not offer count as misses

Also written: answer-quality rates (NO MATCH, unparsed, format-invalid, hallucinated,
UNCERTAIN, truncated), run settings and throughput from ``run_meta.json``, and strata
(single- versus multi-gold abstracts; abstracts whose candidates include several entries of
one gene).

Outputs ``<out_prefix>_summary.json``, ``<out_prefix>_per_tiab.csv`` and a printed table.

    python -m litdd.evaluation.llm_adjudication_eval \\
        --llm_parquet "out/*__llm.parquet" --gold_csv fixture/gold.csv \\
        --pairs_csv fixture/pairs.csv --g2p_csv G2P_DD.csv --out_prefix out/eval --label myrun

Paired comparison of two runs (exact McNemar on per-abstract correctness, paired bootstrap
on the id-micro F1 difference):

    python -m litdd.evaluation.llm_adjudication_eval --compare A_per_tiab.csv B_per_tiab.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random

import pandas as pd

from litdd.evaluation.common import (
    NO_MATCH,
    candidate_ids_from_row,
    g2p_ids_regex,
    g2p_ids_split,
    mcnemar_exact,
    prf,
    wilson_ci,
    x_equivalence_map,
)

logger = logging.getLogger(__name__)


def prf_with_ci(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall, F1 and Wilson 95% intervals for precision and recall."""
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f, 4),
            "precision_ci95": [round(x, 4) for x in wilson_ci(tp, tp + fp)],
            "recall_ci95": [round(x, 4) for x in wilson_ci(tp, tp + fn)]}


def parse_set(v) -> set[str]:
    """Ids in an ``llm_dis_map`` cell."""
    return g2p_ids_regex(v)


def gold_set(v) -> set[str]:
    """Ids in a ``true_g2p_ids`` cell."""
    return g2p_ids_split(v)


def gene_index(g2p_csv: str) -> dict[str, str]:
    """``{g2p id: gene symbol}`` from a G2P export."""
    d = pd.read_csv(g2p_csv, dtype=str)
    d.columns = [c.strip() for c in d.columns]
    return dict(zip(d["g2p id"].str.strip(), d["gene symbol"].str.strip()))


def per_tiab_table(llm: pd.DataFrame, gold: pd.DataFrame, canon: dict[str, str] | None = None,
                   genes: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per fixture abstract with its gold set, predicted set, per-id counts and flags.

    ``canon`` maps equivalent X-linked ids to one id (see ``common.x_equivalence_map``);
    ``genes`` maps ids to gene symbols for the shared-gene stratum."""
    key = "row_id" if "row_id" in llm.columns and "row_id" in gold.columns else "pmid"
    llm = llm.copy()
    llm[key] = llm[key].astype(str)
    gold = gold.copy()
    gold[key] = gold[key].astype(str)
    df = gold.merge(llm, on=key, how="left", suffixes=("", "_llm"), validate="one_to_one")
    missing = df["generated_text"].isna().sum() if "generated_text" in df.columns else len(df)
    if missing:
        logger.warning("%d gold abstracts have no adjudication row", missing)

    rows = []
    canon = canon or {}
    genes = genes or {}
    for r in df.itertuples(index=False):
        g = {canon.get(i, i) for i in gold_set(r.true_g2p_ids)}
        p = {canon.get(i, i) for i in parse_set(getattr(r, "llm_dis_map", None))}
        cands = [canon.get(i, i) for i in candidate_ids_from_row(getattr(r, "candidates", None))]
        cand_genes = [genes.get(i, i) for i in cands]
        bert = int(getattr(r, "bert_predict", 1) or 0)
        p_screen = p if bert == 1 else set()
        raw = getattr(r, "llm_dis_map", None)
        has_row = isinstance(getattr(r, "generated_text", None), str)
        raw_missing = raw is None or (isinstance(raw, float) and math.isnan(raw))
        rows.append({
            key: getattr(r, key), "pmid": r.pmid,
            "n_gold": len(g), "gold": ";".join(sorted(g)), "pred": ";".join(sorted(p)),
            "cand_ids": ";".join(sorted(set(cands))),
            "multi_gold": len(g) > 1,
            "cands_share_gene": len(cand_genes) != len(set(cand_genes)),
            "n_candidates": len(set(cands)),
            "bert_predict": bert,
            "genereviews": bool(getattr(r, "genereviews", False)),
            "exact_correct": p == g,
            "exact_correct_end_to_end": p_screen == g,
            "final_pred": ";".join(sorted(p_screen)),
            "id_tp": len(p & g), "id_fp": len(p - g), "id_fn": len(g - p),
            "id_tp_screen": len(p_screen & g), "id_fp_screen": len(p_screen - g),
            "id_fn_screen": len(g - p_screen),
            "no_match": (str(raw).upper() == NO_MATCH) if not raw_missing else False,
            "unparsed": has_row and raw_missing,
            "format_invalid": has_row and getattr(r, "answer_format_valid", None) is False,
            "uncertain": has_row and getattr(r, "answer_uncertain", None) is True,
            "hallucinated": getattr(r, "answer_ids_in_candidates", None) is False,
            "truncated": getattr(r, "finish_reason", None) == "length",
            "skipped_no_candidates": getattr(r, "finish_reason", None) == "skipped",
            "no_llm_row": not has_row,
            "gen_tokens": getattr(r, "gen_tokens", None),
            "prompt_tokens": getattr(r, "prompt_tokens", None),
        })
    return pd.DataFrame(rows)


def view_id_micro(t: pd.DataFrame, suffix: str = "") -> dict:
    return prf_with_ci(int(t[f"id_tp{suffix}"].sum()), int(t[f"id_fp{suffix}"].sum()),
                       int(t[f"id_fn{suffix}"].sum()))


def view_tiab_exact(t: pd.DataFrame) -> dict:
    has_gold = t["n_gold"] > 0
    has_pred = t["pred"] != ""
    tp = int((has_gold & t["exact_correct"]).sum())
    fp = int((has_pred & ~t["exact_correct"]).sum())
    fn = int((has_gold & ~t["exact_correct"]).sum())
    tn = int((~has_gold & ~has_pred).sum())
    out = prf_with_ci(tp, fp, fn)
    out["tn"] = tn
    out["accuracy"] = round(float(t["exact_correct"].mean()), 4) if len(t) else float("nan")
    return out


def view_end_to_end_exact(t: pd.DataFrame) -> dict:
    """Exact set match over every abstract: a curated abstract is a true positive when the
    screen passed it and the final set equals the curated set; a curated abstract that fails
    anywhere is a false negative; a non-curated abstract with any mapping is a false positive."""
    has_gold = t["n_gold"] > 0
    final_nonempty = t["final_pred"] != ""
    tp = int((has_gold & t["exact_correct_end_to_end"]).sum())
    fn = int((has_gold & ~t["exact_correct_end_to_end"]).sum())
    fp = int((~has_gold & final_nonempty).sum()
             + (has_gold & final_nonempty & ~t["exact_correct_end_to_end"]).sum())
    tn = int((~has_gold & ~final_nonempty).sum())
    out = prf_with_ci(tp, fp, fn)
    out["tn"] = tn
    out["n_abstracts"] = int(len(t))
    return out


def view_pair_level(t: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Pair-level scoring over the labelled (abstract, id) pairs the gate offered."""
    key = "row_id" if "row_id" in t.columns else "pmid"
    pred_by = {str(getattr(r, key)): set(r.pred.split(";")) - {""} for r in t.itertuples(index=False)}
    pairs = pairs.copy()
    pairs[key] = pairs[key].astype(str)
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
    out = prf_with_ci(tp, fp, fn)
    out["tn"] = tn
    out["labelled_pairs"] = int(len(pairs))
    out["pairs_offered"] = int(len(offered))
    out["positive_pairs_not_offered"] = int(((pairs["label"] == 1) & ~pairs["in_candidates"]).sum())
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
    """Settings and throughput from the ``run_meta.json`` files next to the parquets."""
    metas = []
    for p in paths:
        mp = p.replace("__llm.parquet", "__llm.run_meta.json")
        if os.path.exists(mp):
            with open(mp) as f:
                metas.append(json.load(f))
    if not metas:
        return {}
    m = metas[0]
    keep = {k: m.get(k) for k in ("model", "reasoning_effort", "prompt_file", "context_json",
                                  "temperature", "top_p", "max_tokens", "max_model_len", "seed",
                                  "dtype", "git_sha", "image", "versions", "gpu")}
    keep["rows_per_s"] = m.get("rows_per_s")
    keep["gen_tokens_per_s"] = m.get("gen_tokens_per_s")
    keep["generation_s_total"] = round(sum(x.get("generation_s") or 0 for x in metas), 1)
    keep["shards"] = len(metas)
    return keep


def compare(a_csv: str, b_csv: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two per-abstract tables: exact McNemar on correctness and a
    percentile bootstrap on the id-micro F1 difference."""
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


VIEW_ORDER = ("end_to_end_exact", "tiab_exact", "id_micro", "id_micro_screen",
              "id_micro_screen_positives_only", "tiab_exact_screen_positives_only", "pair_level")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llm_parquet", help="path or glob of *__llm.parquet")
    ap.add_argument("--gold_csv")
    ap.add_argument("--pairs_csv", default=None)
    ap.add_argument("--out_prefix")
    ap.add_argument("--label", default=None)
    ap.add_argument("--g2p_csv", default=None,
                    help="G2P export used for the X-linked equivalence rule and the shared-gene "
                         "stratum; both are skipped when absent")
    ap.add_argument("--compare", nargs=2, metavar=("A_PER_TIAB", "B_PER_TIAB"))
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.compare:
        res = compare(*args.compare)
        print(json.dumps(res, indent=2))
        if args.out_prefix:
            with open(f"{args.out_prefix}_compare.json", "w") as f:
                json.dump(res, f, indent=2)
        return 0

    if not (args.llm_parquet and args.gold_csv and args.out_prefix):
        raise SystemExit("--llm_parquet, --gold_csv and --out_prefix are required")
    paths = sorted(glob.glob(args.llm_parquet)) or [args.llm_parquet]
    llm = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    gold = pd.read_csv(args.gold_csv)
    pairs = pd.read_csv(args.pairs_csv) if args.pairs_csv else None

    canon = x_equivalence_map(args.g2p_csv) if args.g2p_csv else {}
    genes = gene_index(args.g2p_csv) if args.g2p_csv else {}
    t = per_tiab_table(llm, gold, canon=canon, genes=genes)
    screen_pos = t[t["bert_predict"] == 1]
    summary = {
        "label": args.label or os.path.basename(args.out_prefix),
        "llm_parquet": paths, "gold_csv": args.gold_csv,
        "run": load_run_meta(paths),
        "rates": rates(t),
        "end_to_end_exact": view_end_to_end_exact(t),
        "tiab_exact": view_tiab_exact(t),
        "id_micro": view_id_micro(t),
        "id_micro_screen": view_id_micro(t, "_screen"),
        "id_micro_screen_positives_only": view_id_micro(screen_pos),
        "tiab_exact_screen_positives_only": view_tiab_exact(screen_pos),
        "pair_level": view_pair_level(t, pairs) if pairs is not None else None,
        "strata": strata(t),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    t.to_csv(f"{args.out_prefix}_per_tiab.csv", index=False)
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n== {summary['label']}  ({len(t)} abstracts; model {summary['run'].get('model')}; "
          f"effort {summary['run'].get('reasoning_effort')})")
    print(f"{'view':<34}{'P':>8}{'R':>8}{'F1':>8}   tp/fp/fn")
    for name in VIEW_ORDER:
        v = summary[name]
        if v:
            print(f"{name:<34}{v['precision']:>8.4f}{v['recall']:>8.4f}{v['f1']:>8.4f}   "
                  f"{v['tp']}/{v['fp']}/{v['fn']}")
    r = summary["rates"]
    print(f"rates: no_match {r['no_match_rate']:.3f}  unparsed {r['unparsed_rate']:.3f}  "
          f"format_invalid {r['format_invalid_rate']:.3f}  hallucinated {r['hallucinated_rate']:.3f}  "
          f"uncertain {r['uncertain_rate']:.3f}  truncated {r['truncated_rate']:.3f}  "
          f"no_llm_row {r['no_llm_row_rate']:.3f}  skipped {r['skipped_no_candidates_rate']:.3f}  "
          f"gen_tokens mean/p95 {r.get('gen_tokens_mean')}/{r.get('gen_tokens_p95')}")
    for k, v in summary["strata"].items():
        if v["id_micro"]:
            print(f"  stratum {k:<22} n={v['n_tiabs']:<5} exact_acc {v['exact_accuracy']:.4f}  "
                  f"id_micro F1 {v['id_micro']['f1']:.4f}")
    print(f"wrote {args.out_prefix}_summary.json / _per_tiab.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
