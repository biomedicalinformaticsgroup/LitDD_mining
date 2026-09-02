#!/usr/bin/env python3
"""Score the completed precision audit and inter-annotator exercises (Reviewer 2 B1 /
R2-A1, Reviewer 3 R3.10).

Reads the annotator-filled worksheets + hidden keys produced by sample_audit.py and
sample_trainlabel_iaa.py, and reports:

  1. Deployed-corpus precision overall and per stratum (confidence / recency /
     disease_volume / gene_multiplicity), with Wilson 95% confidence intervals, and the
     implied false-positive count at the full corpus size (R2-P1/P2).
  2. Error-category breakdown among incorrect mappings, incl. single- vs multi-DDG2P
     genes (feeds R2-D1/D2).
  3. Cohen's kappa between annotators A and B on the audit overlap subset (R2-A1).
  4. Cohen's kappa between a second annotator and the original training labels (R3.10).

Each section is skipped (with a message) if its worksheet has not been annotated yet, so
this can be run as a dry check before annotation. Uses only numpy/pandas.

Outputs a console summary and CSVs under the audit dir (suitable as Source Data).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

STRATA = ["confidence", "recency", "disease_volume", "gene_multiplicity"]


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


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
    print("\n=== Deployed-corpus precision (R2-P1/P2) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nOverall precision {overall['precision']:.3f} "
          f"(95% CI {overall['ci95_low']:.3f}-{overall['ci95_high']:.3f}); "
          f"implied false positives at corpus size {corpus_n:,}: "
          f"~{int(round(fp_rate * corpus_n)):,}")

    # Post-cutoff precision (R3.1): does precision hold on abstracts the LLM could not
    # have memorised (published at/after the model's knowledge cutoff)?
    #
    # Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B. It is the Qwen2.5-14B base
    # (config.json: qwen2) with DeepSeek-R1 reasoning distilled on top — the distillation
    # adds reasoning traces, not a knowledge refresh, so the binding *knowledge* cutoff is
    # Qwen2.5-14B's pretraining cutoff of ~end of 2023 (Dec 2023). Publications from 2024
    # onward are therefore post-cutoff (cutoff_year=2024). (The full DeepSeek-R1 model is
    # 2024-07; our distill is Qwen2.5-based, so 2024 is correct. Use cutoff_year=2025 for an
    # even-safer margin that is post both dates.)
    if cutoff_year is not None and "year" in df.columns:
        yr = pd.to_numeric(df["year"], errors="coerce")
        post = _precision_for(df, yr >= cutoff_year)
        pre = _precision_for(df, yr < cutoff_year)
        print(f"\n=== Post-cutoff precision (R3.1), cutoff={cutoff_year} ===")
        print(f"  post-cutoff (>= {cutoff_year}): precision {post['precision']:.3f} "
              f"(95% CI {post['ci95_low']:.3f}-{post['ci95_high']:.3f}), n={post['n_judged']}")
        print(f"  pre-cutoff  (<  {cutoff_year}): precision {pre['precision']:.3f} "
              f"(95% CI {pre['ci95_low']:.3f}-{pre['ci95_high']:.3f}), n={pre['n_judged']}")
        print("  (similar precision either side argues against memorisation/contamination "
              "inflating the result.)")

    # Error categories among incorrect, by gene multiplicity (R2-D1/D2)
    inc = df[df["verdict"].astype(str).str.strip().str.lower() == "incorrect"]
    if len(inc):
        tab = pd.crosstab(inc["error_category"].fillna("unspecified"),
                          inc.get("gene_multiplicity", pd.Series(["?"] * len(inc))))
        tab.to_csv(audit_dir / "error_categories.csv")
        print("\n=== Error categories among incorrect (R2-D1/D2) ===")
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
    print(f"\n=== Audit inter-annotator agreement (R2-A1), n={len(pairs)} ===")
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
    print(f"\n=== Training-label inter-annotator agreement (R3.10), n={len(df)} ===")
    print(f"  raw agreement {agree:.3f} | Cohen's kappa {k:.3f}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit_dir", default="revision/precision_audit")
    ap.add_argument("--corpus_n", type=int, default=68705,
                    help="Deployed corpus size, for the implied false-positive count")
    ap.add_argument("--cutoff_year", type=int, default=None,
                    help="If set, report precision for records at/after this year "
                         "(LLM-contamination check, R3.1). Use 2024: the model "
                         "(DeepSeek-R1-Distill-Qwen-14B, Qwen2.5-14B base) has a knowledge "
                         "cutoff ~Dec 2023, so 2024+ is post-cutoff. 2025 = extra-safe margin.")
    return ap.parse_args()


def main():
    args = parse_args()
    d = Path(args.audit_dir)
    score_precision(d, args.corpus_n, args.cutoff_year)
    score_audit_iaa(d)
    score_trainlabel_iaa(d)


if __name__ == "__main__":
    main()
