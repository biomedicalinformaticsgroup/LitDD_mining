#!/usr/bin/env python3
"""Adjudication performance on genes with several DDG2P entries (allelic series; R2-D2).

For each arm, a TIAB is *multi-entry* when the gene of (any of) its curated entries has more
than one entry in the G2P export THAT ARM USED -- the original pipeline ran on the 2025-02-15
export, the revised one on 2026-06-24, and the entry counts differ, so each arm is stratified
by its own panel. A paired comparison is then made on the TIABs that are multi-entry under
BOTH panels, so the two arms are judged on the same disambiguation cases.

Per stratum (single-entry / multi-entry gene; all gold TIABs and screen-positive gold TIABs):
per-(TIAB, entry) micro P/R/F1, exact-set accuracy, and the error types that matter for an
allelic series: wrong sibling (an entry of the right gene, but the wrong one), extra sibling
(right entry plus another of the same gene), other gene, NO MATCH, not reached (dropped
upstream).

    python litdd/evaluation/llm_multientry_analysis.py \\
        --arm original=revision/llm_eval/runs/deepseek_deployed/eval_per_tiab.csv:$REF/G2P_DD_2025-02-15.csv \\
        --arm new=revision/llm_eval/eval_2026/gptoss_2026_gated/eval_per_tiab.csv:revision/G2P_DD_2026-06-24.csv \\
        --out_csv revision/llm_eval/eval_2026/multientry.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from litdd.evaluation.compare_models import mcnemar_exact  # noqa: E402
from litdd.threads import load_g2p  # noqa: E402


def gene_index(g2p_csv: str) -> tuple[dict[str, str], dict[str, int]]:
    """{g2p_id: gene symbol}, {gene symbol: number of entries} for one export."""
    df = load_g2p(g2p_csv)
    idc = "g2p id" if "g2p id" in df.columns else "g2p_id"
    gc = "gene symbol" if "gene symbol" in df.columns else "gene_symbol"
    df = df.drop_duplicates(idc)
    id2gene = dict(zip(df[idc].astype(str), df[gc].astype(str)))
    counts = df[gc].astype(str).value_counts().to_dict()
    return id2gene, counts


def split_ids(v) -> set[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return set()
    return {x for x in str(v).split(";") if x and x != "nan"}


def classify(gold: set[str], pred: set[str], id2gene: dict[str, str], reached: bool) -> str:
    if not reached:
        return "not_reached"
    if pred == gold:
        return "correct"
    if not pred:
        return "no_match"
    gold_genes = {id2gene.get(g, "?") for g in gold}
    same_gene_extra = {p for p in pred - gold if id2gene.get(p, "?") in gold_genes}
    if pred & gold:
        return "extra_sibling" if pred - gold and pred - gold == same_gene_extra else "partial_other"
    if pred and all(id2gene.get(p, "?") in gold_genes for p in pred):
        return "wrong_sibling"
    return "other_gene"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def analyse(name: str, per_tiab: pd.DataFrame, id2gene: dict, counts: dict) -> tuple[pd.DataFrame, list[dict]]:
    t = per_tiab.copy()
    t["row_id"] = t["row_id"].astype(str)
    t = t[t["n_gold"] > 0].copy()
    t["gold_set"] = t["gold"].map(split_ids)
    t["pred_set"] = t["pred"].map(split_ids)
    t["gold_genes"] = t["gold_set"].map(lambda s: {id2gene.get(g, "?") for g in s})
    t["max_entries"] = t["gold_genes"].map(lambda gs: max((counts.get(g, 0) for g in gs), default=0))
    t["multi_entry"] = t["max_entries"] > 1
    t["reached"] = ~t["no_llm_row"].astype(bool) if "no_llm_row" in t.columns else True
    t["error_type"] = [classify(g, p, id2gene, r) for g, p, r in zip(t["gold_set"], t["pred_set"], t["reached"])]
    rows = []
    for subset_name, mask in (("all_gold", pd.Series(True, index=t.index)),
                              ("screen_positive", t["bert_predict"] == 1)):
        for stratum, smask in (("multi_entry_gene", t["multi_entry"]), ("single_entry_gene", ~t["multi_entry"])):
            s = t[mask & smask]
            tp = int(sum(len(g & p) for g, p in zip(s["gold_set"], s["pred_set"])))
            fp = int(sum(len(p - g) for g, p in zip(s["gold_set"], s["pred_set"])))
            fn = int(sum(len(g - p) for g, p in zip(s["gold_set"], s["pred_set"])))
            p, r, f = prf(tp, fp, fn)
            counts_e = s["error_type"].value_counts().to_dict()
            rows.append({"arm": name, "subset": subset_name, "stratum": stratum, "n_tiabs": len(s),
                         "n_gold_ids": int(s["n_gold"].sum()),
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
                         "exact_accuracy": round(float((s["error_type"] == "correct").mean()), 4) if len(s) else None,
                         **{f"err_{k}": counts_e.get(k, 0) for k in
                            ("correct", "wrong_sibling", "extra_sibling", "partial_other", "other_gene",
                             "no_match", "not_reached")}})
    return t, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=per_tiab.csv:g2p_csv[:pairs.csv] (the export that arm used; "
                         "pairs.csv enables the labelled-negative sibling specificity)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    tables, all_rows, spec_rows = {}, [], []
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        per_tiab_csv, g2p_csv = parts[0], parts[1]
        pairs_csv = parts[2] if len(parts) > 2 else None
        id2gene, counts = gene_index(g2p_csv)
        per_tiab = pd.read_csv(per_tiab_csv)
        t, rows = analyse(name, per_tiab, id2gene, counts)
        tables[name] = t
        all_rows += rows
        if pairs_csv:
            # Sibling specificity: clinician-labelled NEGATIVE pairs whose entry belongs to a
            # multi-entry gene (these live on all-negative abstracts -- the split keeps only
            # positive pairs on abstracts that have any positive, so precision of extra entries
            # on positive abstracts is NOT measurable from labels; this is).
            full = per_tiab.copy()
            full["row_id"] = full["row_id"].astype(str)
            full = full.set_index("row_id")
            pairs = pd.read_csv(pairs_csv)
            pairs["row_id"] = pairs["row_id"].astype(str)
            neg = pairs[(pairs["label"] == 0)
                        & pairs["g2p_id"].map(lambda x: counts.get(id2gene.get(x, "?"), 0) > 1)]
            for subset_name, need_screen in (("all_tiabs", False), ("screen_positive", True)):
                n = fp = 0
                for r in neg.itertuples(index=False):
                    if r.row_id not in full.index:
                        continue
                    if need_screen and int(full.loc[r.row_id, "bert_predict"]) != 1:
                        continue
                    n += 1
                    fp += r.g2p_id in split_ids(full.loc[r.row_id, "pred"])
                spec_rows.append({"arm": name, "subset": subset_name,
                                  "labelled_negative_sibling_pairs": n, "wrongly_mapped": fp,
                                  "false_mapping_rate": round(fp / n, 4) if n else None})
    out = pd.DataFrame(all_rows)
    out.to_csv(args.out_csv, index=False)
    if spec_rows:
        spec = pd.DataFrame(spec_rows)
        spec.to_csv(args.out_csv.replace(".csv", "_sibling_specificity.csv"), index=False)
        print("\nSibling specificity on labelled-negative pairs of multi-entry genes:")
        print(spec.to_string(index=False))
    pd.set_option("display.width", 250)
    cols = ["arm", "subset", "stratum", "n_tiabs", "precision", "recall", "f1", "exact_accuracy",
            "err_wrong_sibling", "err_extra_sibling", "err_no_match", "err_other_gene", "err_not_reached"]
    print(out[cols].to_string(index=False))

    # Paired comparison on TIABs that are multi-entry under BOTH panels (first arm = reference)
    names = list(tables)
    ref = tables[names[0]]
    for other in names[1:]:
        o = tables[other]
        common = sorted(set(ref.loc[ref["multi_entry"], "row_id"]) & set(o.loc[o["multi_entry"], "row_id"]))
        a = ref.set_index("row_id").loc[common]
        b = o.set_index("row_id").loc[common]
        ca = (a["error_type"] == "correct").astype(int).tolist()
        cb = (b["error_type"] == "correct").astype(int).tolist()
        x, y, p = mcnemar_exact([1] * len(common), ca, cb)

        def micro(s):
            return prf(int(sum(len(g & q) for g, q in zip(s["gold_set"], s["pred_set"]))),
                       int(sum(len(q - g) for g, q in zip(s["gold_set"], s["pred_set"]))),
                       int(sum(len(g - q) for g, q in zip(s["gold_set"], s["pred_set"]))))
        pa, ra, fa = micro(a)
        pb, rb, fb = micro(b)
        print(f"\nPAIRED on {len(common)} TIABs multi-entry in both panels: {names[0]} vs {other}")
        print(f"  exact accuracy {sum(ca)/len(ca):.4f} vs {sum(cb)/len(cb):.4f}; "
              f"{names[0]}-right/{other}-wrong {x}, {names[0]}-wrong/{other}-right {y}, McNemar p={p:.4g}")
        print(f"  P/R/F1 {pa:.3f}/{ra:.3f}/{fa:.3f} vs {pb:.3f}/{rb:.3f}/{fb:.3f}")
        print("  error types", names[0], a["error_type"].value_counts().to_dict())
        print("  error types", other, b["error_type"].value_counts().to_dict())
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
