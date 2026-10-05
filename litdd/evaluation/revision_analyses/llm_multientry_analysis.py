#!/usr/bin/env python3
"""Adjudication performance on genes with several G2P entries (allelic series).

For each arm, an abstract is multi-entry when the gene of any of its curated entries has more
than one entry in the G2P export that arm used, so each arm is stratified by its own panel.
A paired comparison is made on the abstracts that are multi-entry under every panel.

Per stratum (single-entry or multi-entry gene; all gold abstracts and screen-positive gold
abstracts): per-(abstract, entry) micro precision, recall and F1, exact-set accuracy, and the
error types of an allelic series: wrong sibling (an entry of the right gene, but the wrong
one), extra sibling (the right entry plus another of the same gene), other gene, NO MATCH,
not reached (dropped upstream).

    python -m litdd.evaluation.revision_analyses.llm_multientry_analysis \\
        --arm a=runs/a/eval_per_tiab.csv:G2P_a.csv --arm b=runs/b/eval_per_tiab.csv:G2P_b.csv \\
        --out_csv multientry.csv
"""
from __future__ import annotations

import argparse

import pandas as pd

from litdd.evaluation.common import g2p_ids_split, mcnemar_exact, prf
from litdd.threads import load_g2p


def gene_index(g2p_csv: str) -> tuple[dict[str, str], dict[str, int]]:
    """``{g2p_id: gene symbol}`` and ``{gene symbol: number of entries}`` for one export."""
    df = load_g2p(g2p_csv)
    idc = "g2p id" if "g2p id" in df.columns else "g2p_id"
    gc = "gene symbol" if "gene symbol" in df.columns else "gene_symbol"
    df = df.drop_duplicates(idc)
    id2gene = dict(zip(df[idc].astype(str), df[gc].astype(str)))
    counts = df[gc].astype(str).value_counts().to_dict()
    return id2gene, counts


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


def _micro(s: pd.DataFrame) -> tuple[float, float, float]:
    tp = int(sum(len(g & q) for g, q in zip(s["gold_set"], s["pred_set"])))
    fp = int(sum(len(q - g) for g, q in zip(s["gold_set"], s["pred_set"])))
    fn = int(sum(len(g - q) for g, q in zip(s["gold_set"], s["pred_set"])))
    return prf(tp, fp, fn)


def analyse(name: str, per_tiab: pd.DataFrame, id2gene: dict, counts: dict) -> tuple[pd.DataFrame, list[dict]]:
    t = per_tiab.copy()
    t["row_id"] = t["row_id"].astype(str)
    t = t[t["n_gold"] > 0].copy()
    t["gold_set"] = t["gold"].map(g2p_ids_split)
    t["pred_set"] = t["pred"].map(g2p_ids_split)
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
            p, r, f = _micro(s)
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
            # Labelled negative pairs whose entry belongs to a multi-entry gene measure how
            # often a sibling entry is wrongly mapped.
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
                    fp += r.g2p_id in g2p_ids_split(full.loc[r.row_id, "pred"])
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
        pa, ra, fa = _micro(a)
        pb, rb, fb = _micro(b)
        print(f"\nPaired on {len(common)} abstracts multi-entry in both panels: {names[0]} vs {other}")
        print(f"  exact accuracy {sum(ca)/len(ca):.4f} vs {sum(cb)/len(cb):.4f}; "
              f"{names[0]}-right/{other}-wrong {x}, {names[0]}-wrong/{other}-right {y}, McNemar p={p:.4g}")
        print(f"  P/R/F1 {pa:.3f}/{ra:.3f}/{fa:.3f} vs {pb:.3f}/{rb:.3f}/{fb:.3f}")
        print("  error types", names[0], a["error_type"].value_counts().to_dict())
        print("  error types", other, b["error_type"].value_counts().to_dict())
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
