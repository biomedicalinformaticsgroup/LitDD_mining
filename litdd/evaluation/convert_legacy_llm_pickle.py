#!/usr/bin/env python3
"""Convert the original paper's LLM-run pickle into the current ``__llm.parquet`` contract.

The deployed DeepSeek-R1-Distill-Qwen-14B adjudication of the annotated test split was
saved by ``train_test/evaluate_2_llm.py`` as ``anno_cross_<model>_df.pkl`` (one row per
labelled pair, columns tiab / top_5_cross / llm_prompt / generated_text / llm_dis_map).
This re-keys it onto the per-TIAB fixture from ``build_llm_eval_shards.py`` and re-parses
the generations with the current parser, so the deployed baseline can be scored by
``llm_adjudication_eval.py`` on exactly the same footing as the new runs -- without
re-running the model.

    python litdd/evaluation/convert_legacy_llm_pickle.py \\
        --pkl $REF/anno_cross_DeepSeek-R1-Distill-Qwen-14B_df.pkl \\
        --shard revision/llm_eval/annotated_2025/shards/annotated_test.parquet \\
        --out_dir revision/llm_eval/runs/deepseek_deployed --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "litdd", "pipeline"))

import llm_map  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--shard", required=True, help="annotated_test.parquet from build_llm_eval_shards")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    args = ap.parse_args()

    legacy = pd.read_pickle(args.pkl)
    shard = pd.read_parquet(args.shard)
    need = {"tiab", "generated_text", "llm_prompt"}
    if not need <= set(legacy.columns):
        raise SystemExit(f"[ERROR] {args.pkl} lacks {need - set(legacy.columns)}")

    # The paper ran one prompt per labelled PAIR, so a TIAB with several pairs was generated
    # several times (temperature 0, so normally identical). Keep the first; count the others.
    g = legacy.groupby("tiab", sort=False)
    first = g.first()
    n_multi = int((g.size() > 1).sum())
    n_divergent = int(g["generated_text"].nunique().gt(1).sum())

    df = shard.merge(first[["llm_prompt", "generated_text"]], left_on="tiab", right_index=True,
                     how="left", validate="one_to_one")
    missing = int(df["generated_text"].isna().sum())
    if missing:
        print(f"[WARN] {missing} fixture TIABs absent from the legacy pickle")

    df["topk_cross_lgmde"] = df["top5_cross"].apply(llm_map.to_labels)
    df["top_5_cross_lgmde"] = df["topk_cross_lgmde"]
    allowed = df["topk_cross_lgmde"].apply(llm_map.candidate_ids)
    raws = df["generated_text"].apply(lambda t: llm_map.extract_last_answer(t) if isinstance(t, str) else None)
    parsed = pd.DataFrame([llm_map.parse_answer(r, a) for r, a in zip(raws, allowed)])
    df["llm_answer_raw"] = raws
    for c in parsed.columns:
        df[c] = parsed[c]
    df["finish_reason"] = None
    df["prompt_tokens"] = None
    df["gen_tokens"] = None

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.shard))[0]
    out = os.path.join(args.out_dir, f"{stem}__llm.parquet")
    df.to_parquet(out, index=False)
    meta = {
        "stage": "llm_map (legacy conversion)", "source": os.path.abspath(args.pkl),
        "model": args.model, "use_chat_template": False, "reasoning_effort": None,
        "threads": "vanilla", "temperature": 0.0, "top_p": 1.0, "max_tokens": 10000,
        "note": "generations from train_test/evaluate_2_llm.py (raw prompt string, vLLM "
                "generate); re-parsed with the current parser",
        "rows_total": int(len(df)), "rows_missing_generation": missing,
        "legacy_tiabs_with_multiple_generations": n_multi,
        "legacy_tiabs_with_divergent_generations": n_divergent,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out.replace("__llm.parquet", "__llm.run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"[Info] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
