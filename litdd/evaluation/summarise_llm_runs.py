#!/usr/bin/env python3
"""One row per LLM-evaluation run: settings, throughput and every metric view.

    python litdd/evaluation/summarise_llm_runs.py --runs_dir revision/llm_eval/runs \\
        --out_csv revision/llm_eval/summary.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

VIEWS = ("paper_legacy", "id_micro_screen_positives_only", "tiab_exact_screen_positives_only",
         "id_micro_screen_gated", "id_micro_gated", "id_micro", "pair_level")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.runs_dir, "*", "eval_summary.json"))):
        with open(path) as f:
            s = json.load(f)
        run = s.get("run", {})
        r = {"run": s["label"], "model": run.get("model"), "threads": run.get("threads"),
             "reasoning_effort": run.get("reasoning_effort"),
             "chat_template": run.get("use_chat_template"),
             "temperature": run.get("temperature"), "max_tokens": run.get("max_tokens"),
             "max_model_len": run.get("max_model_len"),
             "rows_per_s": run.get("rows_per_s"), "gen_tokens_per_s": run.get("gen_tokens_per_s"),
             "generation_s": run.get("generation_s_total"), "gpu": run.get("gpu"),
             "vllm": (run.get("versions") or {}).get("vllm"),
             "n_tiabs": s["rates"]["n_tiabs"]}
        for k in ("no_match_rate", "unparsed_rate", "format_invalid_rate", "hallucinated_rate",
                  "uncertain_rate", "truncated_rate", "no_llm_row_rate", "gen_tokens_mean",
                  "gen_tokens_p95", "prompt_tokens_mean"):
            r[k] = s["rates"].get(k)
        for v in VIEWS:
            x = s.get(v)
            if x:
                for m in ("precision", "recall", "f1", "tp", "fp", "fn"):
                    r[f"{v}__{m}"] = x.get(m)
        for st in ("multi_gold", "cands_share_gene"):
            x = s["strata"].get(st, {})
            r[f"stratum_{st}__n"] = x.get("n_tiabs")
            r[f"stratum_{st}__exact_accuracy"] = x.get("exact_accuracy")
            r[f"stratum_{st}__id_micro_f1"] = (x.get("id_micro") or {}).get("f1")
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    cols = ["run", "threads", "reasoning_effort", "rows_per_s", "no_match_rate", "truncated_rate",
            "paper_legacy__f1", "id_micro_screen_positives_only__precision",
            "id_micro_screen_positives_only__recall", "id_micro_screen_positives_only__f1",
            "id_micro__precision"]
    pd.set_option("display.width", 250)
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"wrote {args.out_csv} ({len(df)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
