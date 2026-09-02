#!/usr/bin/env python3
"""One clean ledger of every LLM-stage ablation: re-score all runs at their design cutoff.

The evaluator's historic default (--score_cutoff 0.9) applied a post-hoc score gate inside
the end_to_end_exact view even for arms whose adopted design has no threshold; this script
re-scores every run with the cutoff its design actually specifies (0 for all no-threshold
arms; 0.9 for the original pipeline, whose design gated after the LLM) and writes one row
per run with the confusion matrix and P/R/F1, plus provenance columns.

    python litdd/evaluation/build_ablation_ledger.py --out_csv supplementary/llm_ablation_ledger.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# run -> (fixture dir, split, design cutoff, description)
F_T25 = "revision/llm_eval/annotated_2025"
F_T25D = "revision/llm_eval/annotated_2025_deployedscreen"
F_T26 = "revision/llm_eval/annotated_2026"
F_DEV = "revision/llm_eval/dev_train_2026"
F_EXT = "revision/llm_eval/external_2026"
RUNS = {
    # original pipeline and hybrids (2025 panel; design = gene check + 0.9 gate after the LLM)
    "deepseek_deployed": (F_T25D, "test", 0.9, "ORIGINAL: old screen(verified), old CE top-5, DeepSeek-R1-14B, 0.9 gate after"),
    "gptoss_vanilla": (F_T25D, "test", 0.9, "original candidates + GPT-OSS (ablation: LLM swap only)"),
    "deepseek_chat": (F_T25, "test", 0.9, "DeepSeek re-run through the chat template (control)"),
    "gptoss_context": (F_T25, "test", 0.9, "contextualised threads"),
    "gptoss_vanilla_low": (F_T25, "test", 0.9, "reasoning effort low"),
    "gptoss_vanilla_high": (F_T25, "test", 0.9, "reasoning effort high (worse: truncation)"),
    # old-gate revised arms (superseded by the TIAB gate; kept for the ledger)
    "gptoss_2026_top5": (F_T26, "test", 0.0, "v2.0 CE full-panel top-5 -> LLM (original order, new models)"),
    "gptoss_2026_gated_tnone": (F_T26, "test", 0.0, "OLD gene gate -> v2.0 CE -> all entries -> LLM"),
    "gptoss_2026_gated": (F_T26, "test", None, "OLD gate, CE>=0.9 shown to LLM (pre-gated)"),
    "gptoss_2026_gated_t03": (F_T26, "test", None, "OLD gate, CE>=0.3 shown"),
    "gptoss_2026_gated_t05": (F_T26, "test", None, "OLD gate, CE>=0.5 shown"),
    "gptoss_2026_gated_t07": (F_T26, "test", None, "OLD gate, CE>=0.7 shown"),
    "gptoss_2026_gated_t08": (F_T26, "test", None, "OLD gate, CE>=0.8 shown"),
    "gptoss_2026_gated_t095": (F_T26, "test", None, "OLD gate, CE>=0.95 shown"),
    "gptoss_2026_gated_k1": (F_T26, "test", 0.0, "OLD gate, top-1 by CE score"),
    "gptoss_2026_gated_k2": (F_T26, "test", 0.0, "OLD gate, top-2"),
    "gptoss_2026_gated_k3": (F_T26, "test", 0.0, "OLD gate, top-3"),
    "gptoss_2026_gated_k5": (F_T26, "test", 0.0, "OLD gate, top-5"),
    "gptoss_2026_gated_singlebest": (F_T26, "test", 0.0, "prompt: single best entry per gene"),
    "gptoss_2026_gated_fewshot": (F_T26, "test", 0.0, "prompt: two train-split worked examples"),
    "gptoss_2026_gated_bygene": (F_T26, "test", 0.0, "layout: one candidate block per gene"),
    "gptoss_2026_gated_bygene_singlebest": (F_T26, "test", 0.0, "by-gene layout + single-best prompt"),
    "gptoss_2026_gatedfb_tnone": (F_T26, "test", 0.0, "OLD gate + symbol fallback -> all entries"),
    # corrected TIAB gate arms
    "gptoss_test_gatedtb_tnone": (F_T26, "test", 0.0, "FINAL: screen -> TIAB gene gate -> [CE scores unused] -> all entries -> LLM"),
    "gptoss_test_gatedtb_dname": (F_T26, "test", 0.0, "disease-name-priority prompt (dev-selected, test-neutral)"),
    "gptoss_test_direct_llm": (F_T26, "test", 0.0, "DIRECT: TIAB gene gate -> all entries -> LLM (no screen, no CE) [score without screen via gold_noscreen]"),
    # dev split (model-selection sandbox)
    "gptoss_dev_gatedfb_tnone": (F_DEV, "dev", 0.0, "dev baseline, old gate + fallback"),
    "gptoss_dev_gatedfb_json": (F_DEV, "dev", 0.0, "structured per-gene JSON output v1"),
    "gptoss_dev_gatedfb_json2": (F_DEV, "dev", 0.0, "structured JSON v2 (multi-disorder fixed)"),
    "gptoss_dev_gatedfb_scores": (F_DEV, "dev", 0.0, "cross-encoder score shown per candidate"),
    "gptoss_dev_gatedfb_hpo": (F_DEV, "dev", 0.0, "HPO phenotype terms per candidate (all)"),
    "gptoss_dev_gatedtb_tnone": (F_DEV, "dev", 0.0, "dev baseline, TIAB gene gate"),
    "gptoss_dev_gatedtb_hpomulti": (F_DEV, "dev", 0.0, "HPO terms only for multi-entry genes"),
    "gptoss_dev_gatedtb_dname": (F_DEV, "dev", 0.0, "disease-name-priority prompt"),
    "gptoss_dev_gatedtb_zyg": (F_DEV, "dev", 0.0, "zygosity inference from pedigree cues (v1)"),
    "gptoss_dev_gatedtb_zyg2": (F_DEV, "dev", 0.0, "zygosity v2: +variant patterns, deletion rule, AR overrules name"),
    "deployed_no_ce": (F_T26, "test", 0.0, "DEPLOYED: screen -> TIAB gene gate -> all candidates -> GPT-OSS-20B (no cross-encoder, no threshold)"),
    # candidate-presentation, decoding, model and prompt-family ablations (deployed cascade)
    "repl_context": (F_T26, "test", 0.0, "contextualised candidate threads"),
    "repl_temp07": (F_T26, "test", 0.0, "temperature 0.7 / top_p 0.95"),
    "repl_effort_low": (F_T26, "test", 0.0, "reasoning effort low"),
    "repl_effort_high": (F_T26, "test", 0.0, "reasoning effort high"),
    "repl_barebone": (F_T26, "test", 0.0, "minimal (barebone) prompt"),
    "repl_barebone_ctx": (F_T26, "test", 0.0, "minimal prompt + contextualised threads"),
    "repl_pe_v1": (F_T26, "test", 0.0, "per-candidate binary adjudication, prompt v1"),
    "repl_pe_v5": (F_T26, "test", 0.0, "per-candidate binary adjudication, prompt v5"),
    "repl_pe_v10": (F_T26, "test", 0.0, "per-candidate binary adjudication, prompt v10"),
    "repl_sc3": (F_T26, "test", 0.0, "self-consistency, 3 samples, majority answer set"),
    "repl_sc5": (F_T26, "test", 0.0, "self-consistency, 5 samples, majority answer set"),
    "repl_deepseek": (F_T26, "test", 0.0, "model: DeepSeek-R1-Distill-Qwen-14B"),
    "repl_qwen3": (F_T26, "test", 0.0, "model: Qwen3-30B-A3B-Instruct-2507"),
    # external held-out
    "gptoss_external_gatedtb_tnone": (F_EXT, "external", 0.0, "full pipeline on held-out curated sets"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--eval_dir", default="revision/llm_eval/eval_ledger")
    args = ap.parse_args()
    rows = []
    for run, (fix, split, cutoff, desc) in RUNS.items():
        src = "gptoss_test_direct_llm" if run == "deployed_no_ce" else run
        pq = glob.glob(f"revision/llm_eval/runs/{src}/*__llm.parquet")
        if not pq:
            print(f"[skip] {run}: no parquet")
            continue
        gold = f"{fix}/gold_noscreen.csv" if run == "gptoss_test_direct_llm" else f"{fix}/gold.csv"
        pairs = f"{fix}/pairs_full.csv" if os.path.exists(f"{fix}/pairs_full.csv") else f"{fix}/pairs.csv"
        out_prefix = os.path.join(args.eval_dir, run, "eval")
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
        cmd = [sys.executable, "litdd/evaluation/llm_adjudication_eval.py",
               "--llm_parquet", f"revision/llm_eval/runs/{src}/*__llm.parquet",
               "--gold_csv", gold, "--pairs_csv", pairs,
               "--out_prefix", out_prefix, "--label", run]
        if cutoff is not None:
            cmd += ["--score_cutoff", str(cutoff)]
        else:
            cmd += ["--score_cutoff", "0"]      # predictions were gated before the LLM
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[FAIL] {run}: {r.stderr[-300:]}")
            continue
        s = json.load(open(out_prefix + "_summary.json"))
        e = s["end_to_end_exact"]
        meta = s.get("run", {})
        rows.append({
            "run": run, "split": split, "description": desc,
            "design_cutoff": "pre-LLM" if cutoff is None else cutoff,
            "TP": e["tp"], "FP": e["fp"], "FN": e["fn"], "TN": e["tn"],
            "precision": e["precision"], "recall": e["recall"], "f1": e["f1"],
            "no_match_rate": s["rates"]["no_match_rate"],
            "multi_gold_exact": s["strata"]["multi_gold"]["exact_accuracy"],
            "share_gene_exact": s["strata"]["cands_share_gene"]["exact_accuracy"],
            "rows_per_s": meta.get("rows_per_s"),
            "source": "LitDD-BERT test/dev/external fixtures, corrected labels, end-to-end exact-set scoring at the design cutoff",
        })
        print(f"[ok] {run}: {e['precision']:.4f}/{e['recall']:.4f}/{e['f1']:.4f}")
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
