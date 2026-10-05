#!/usr/bin/env python3
"""Score the completed precision audit and the inter-annotator exercises.

Reads the annotated worksheets and the keys written by ``sample_audit.py`` and
``sample_trainlabel_iaa.py`` and reports:

  1. precision of the released map overall and per stratum (recency, disease_volume,
     gene_multiplicity) with Wilson 95% intervals, and the implied false-positive count at
     the corpus size;
  2. error categories among incorrect mappings, by gene multiplicity;
  3. Cohen's kappa between annotators A and B on the audit overlap subset;
  4. Cohen's kappa between a second annotator and the original training labels.

A section is skipped with a message when its worksheet has not been annotated. Writes a
console summary and CSVs under the audit directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from litdd.evaluation.common import wilson_ci

STRATA = ["recency", "disease_volume", "gene_multiplicity"]


def cohen_kappa(a, b) -> float:
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _precision_row(label: str, verdicts: pd.Series) -> dict:
    v = verdicts.str.strip().str.lower()
    correct = int((v == "correct").sum())
    incorrect = int((v == "incorrect").sum())
    n = correct + incorrect  # 'uncertain' / blank excluded from precision
    lo, hi = wilson_ci(correct, n)
    return {"stratum": label, "n_judged": n, "correct": correct, "incorrect": incorrect,
            "uncertain": int((v == "uncertain").sum()),
            "precision": correct / n if n else float("nan"),
            "ci95_low": lo, "ci95_high": hi}


def _precision_for(df: pd.DataFrame, mask) -> dict:
    return _precision_row("", df.loc[mask, "verdict"].astype(str))


def score_precision(audit_dir: Path, corpus_n: int, cutoff_year=None):
    ws = audit_dir / "audit_worksheet.csv"
    key = audit_dir / "audit_key.csv"
    if not ws.exists() or not key.exists():
        print("[precision] worksheet/key not found — skipping.")
        return
    w = pd.read_csv(ws)
    if not w["verdict"].astype(str).str.strip().str.lower().isin(["correct", "incorrect", "uncertain"]).any():
        print("[precision] worksheet not annotated yet — skipping.")
        return
    df = w.merge(pd.read_csv(key), on="audit_id", how="left")

    rows = [_precision_row("OVERALL", df["verdict"].astype(str))]
    for col in STRATA:
        for level, sub in df.groupby(col):
            rows.append(_precision_row(f"{col}={level}", sub["verdict"].astype(str)))
    out = pd.DataFrame(rows)
    out.to_csv(audit_dir / "precision_by_stratum.csv", index=False)

    overall = out.iloc[0]
    fp_rate = 1 - overall["precision"]
    print("\n=== Precision of the released map ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOverall precision {overall['precision']:.3f} "
          f"(95% CI {overall['ci95_low']:.3f}-{overall['ci95_high']:.3f}); "
          f"implied false positives at corpus size {corpus_n:,}: "
          f"~{int(round(fp_rate * corpus_n)):,}")

    # Precision on records published at or after the adjudication model's knowledge cutoff,
    # compared with earlier records.
    if cutoff_year is not None and "year" in df.columns:
        yr = pd.to_numeric(df["year"], errors="coerce")
        post = _precision_for(df, yr >= cutoff_year)
        pre = _precision_for(df, yr < cutoff_year)
        print(f"\n=== Precision by publication year relative to cutoff {cutoff_year} ===")
        print(f"  post-cutoff (>= {cutoff_year}): precision {post['precision']:.3f} "
              f"(95% CI {post['ci95_low']:.3f}-{post['ci95_high']:.3f}), n={post['n_judged']}")
        print(f"  pre-cutoff  (<  {cutoff_year}): precision {pre['precision']:.3f} "
              f"(95% CI {pre['ci95_low']:.3f}-{pre['ci95_high']:.3f}), n={pre['n_judged']}")

    # Error categories among incorrect mappings, by gene multiplicity.
    inc = df[df["verdict"].astype(str).str.strip().str.lower() == "incorrect"]
    if len(inc):
        tab = pd.crosstab(inc["error_category"].fillna("unspecified"),
                          inc.get("gene_multiplicity", pd.Series(["?"] * len(inc))))
        tab.to_csv(audit_dir / "error_categories.csv")
        print("\n=== Error categories among incorrect mappings ===")
        print(tab.to_string())


def score_audit_iaa(audit_dir: Path):
    a, b = audit_dir / "audit_worksheet.csv", audit_dir / "audit_worksheet_overlap.csv"
    if not a.exists() or not b.exists():
        print("\n[audit-IAA] worksheets not found — skipping.")
        return
    wa = pd.read_csv(a).set_index("audit_id")["verdict"]
    wb = pd.read_csv(b).set_index("audit_id")["verdict"]
    common = [i for i in wb.index if i in wa.index]
    pairs = [(str(wa[i]).strip().lower(), str(wb[i]).strip().lower()) for i in common]
    pairs = [(x, y) for x, y in pairs if x and y and x != "nan" and y != "nan"]
    if not pairs:
        print("\n[audit-IAA] overlap not annotated by both yet — skipping.")
        return
    k = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
    agree = sum(1 for x, y in pairs if x == y) / len(pairs)
    print(f"\n=== Audit inter-annotator agreement, n={len(pairs)} ===")
    print(f"  raw agreement {agree:.3f} | Cohen's kappa {k:.3f}")


def score_trainlabel_iaa(audit_dir: Path):
    ws, key = audit_dir / "trainlabel_iaa_worksheet.csv", audit_dir / "trainlabel_iaa_key.csv"
    if not ws.exists() or not key.exists():
        print("\n[trainlabel-IAA] worksheet/key not found — skipping.")
        return
    w = pd.read_csv(ws)
    if not w["relevant"].astype(str).str.strip().isin(["0", "1"]).any():
        print("\n[trainlabel-IAA] worksheet not annotated yet — skipping.")
        return
    df = w.merge(pd.read_csv(key), on="iaa_id", how="left")
    df = df[df["relevant"].astype(str).str.strip().str.lower().isin(["0", "1"])]
    b = df["relevant"].astype(str).str.strip().astype(int)
    orig = df["original_label"].astype(int)
    k = cohen_kappa(orig.tolist(), b.tolist())
    agree = (orig.values == b.values).mean()
    print(f"\n=== Training-label inter-annotator agreement, n={len(df)} ===")
    print(f"  raw agreement {agree:.3f} | Cohen's kappa {k:.3f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--corpus_n", type=int, required=True,
                    help="number of mappings in the released map, for the implied false-positive count")
    ap.add_argument("--cutoff_year", type=int, default=None,
                    help="if set, report precision separately for records published at or after "
                         "this year (the adjudication model's knowledge cutoff)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    d = Path(args.audit_dir)
    score_precision(d, args.corpus_n, args.cutoff_year)
    score_audit_iaa(d)
    score_trainlabel_iaa(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
