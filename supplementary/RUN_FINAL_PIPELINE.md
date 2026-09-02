# LitDD — final end-to-end pipeline: where it is and how to run it

**Repo**: `/home/eidf128/eidf128/shared/export/michael/litdd_clean`, branch `cleanup/reviewer-fixes`
(remote `github.com/biomedicalinformaticsgroup/LitDD_mining`).

## The pipeline (as settled 2026-09-02)

    PubMed TIABs
      → 1. screen        litdd/pipeline/bert_predict_vllm.py     HF tmy100000001/LitDD_BERT (main = add20k seed 44)
      → 2. gene gate     litdd/pipeline/gene_candidates.py       PubTator3 TIAB-verified mentions + HGNC names + --symbol_fallback
      → 3. adjudication  litdd/pipeline/llm_map.py               openai/gpt-oss-20b, prompts/original_paper.txt, all candidates, NO score threshold
      → 4. clean         litdd/pipeline/final_data_clean.py      --score_cutoff 0

**The cross-encoder is no longer in the cascade** (the gene gate supplies the candidates;
retrieval recall 1.000 on the test set). `tmy100000001/LitDD_crossencoder` is retained on HF for
reproducing the published pipeline and as an optional ranker/audit signal only.

**Measured on the held-out annotated test split** (2,731 abstracts, 646 curated, exact-set match
end to end): **P 0.840 / R 0.848 / F1 0.844** (TP 547, FP 104, FN 98, TN 2,028).
Per-stage confusion matrices: `supplementary/stage_confusion_matrices.csv`.

## Run it on a new corpus (GPU, k8s)

Container `ghcr.io/biomedicalinformaticsgroup/litdd_mining:sha-b75f6dfe471732b1afde95fc0b0b68023462eada`
(vLLM 0.23.0). Manifest to copy: `revision/llm_cascade_tiabgate_job.yaml`. Models must be staged
into a user-owned HF cache first (pods have no egress): `revision/hf_cache_llm` (GPT-OSS-20B) and
`revision/hf_cache_ce` (only if the optional ranker is wanted).

```bash
# 1. screen  (⚠ use the HF checkpoint id, NOT a local checkpoint dir: vLLM v0.23 pooling
#             returned a constant probability from a local dir — see the note below)
python litdd/pipeline/bert_predict_vllm.py --model tmy100000001/LitDD_BERT \
  --input_dir <parquet shards with pmid,tiab,languages,pubdate> --processed_dir screened/ --dtype float32

# 2. gene gate
python litdd/pipeline/gene_candidates.py \
  --input_parquet screened/bert_positive.parquet \
  --g2p_csv revision/G2P_DD_2026-06-24.csv \
  --gene2pubtator data/reference/gene2pubtator3.gz \
  --gene_info revision/human_gene_info.gz \
  --hgnc data/reference/hgnc_complete_set.txt \
  --symbol_fallback --out_parquet candidates.parquet

# 3. LLM adjudication  (candidates.parquet needs a top5_cross column: either run
#    crossencode.py --candidates_parquet for scored/ordered candidates, or build the
#    placeholder column as in the direct arm — see litdd/evaluation/build_llm_eval_shards.py)
python litdd/pipeline/llm_map.py --shards_dir shards/ --out_dir out/ \
  --llm_model openai/gpt-oss-20b --temperature 0.0 --top_p 1.0 --seed 0 \
  --reasoning_effort medium --max_model_len 32768 --save_every 100000

# 4. clean / gate
python litdd/pipeline/final_data_clean.py --llm_parquet "out/*__llm.parquet" --score_cutoff 0
```

## Evaluate a run

```bash
python litdd/evaluation/llm_adjudication_eval.py \
  --llm_parquet "out/*__llm.parquet" \
  --gold_csv revision/llm_eval/annotated_2026/gold.csv \
  --pairs_csv revision/llm_eval/annotated_2026/pairs_full.csv \
  --score_cutoff 0 --out_prefix out/eval --label myrun
python litdd/evaluation/stage_confusion_matrices.py --run myrun \
  --fixture revision/llm_eval/annotated_2026 --out_csv out/stages.csv
```

Fixtures: `revision/llm_eval/annotated_2026` (test), `dev_train_2026` (development — use this for
any tuning), `external_2026` (held-out curated sets: end-to-end recall 0.937).

## Things a new session must know

* **Labels**: use `data/annotation_corrections.csv` via `litdd/evaluation/apply_annotation_corrections.py`
  and `complete_test_pairs.py`; the raw published annotation is incomplete (dropped sibling
  negatives) and contains three corrected/removed groups (PURA, co-reported pairs, G2P01446).
* **Scoring**: the design has **no score threshold** — always pass `--score_cutoff 0`. The
  evaluator's 0.9 default is only for reproducing the original pipeline.
* **Screen bug**: `bert_predict_vllm.py` + a *local* add20k checkpoint directory under vLLM 0.23
  produced a constant probability for every abstract. Use the HF id, and sanity-check the
  positive rate (~0.5% of random PubMed) before committing a corpus run.
* **G2P version**: everything current uses `revision/G2P_DD_2026-06-24.csv`. The 2025-02-15 export
  is only for reproducing the published pipeline; never mix versions within one evaluation.
* Results ledger: `supplementary/llm_ablation_ledger.csv`; reviewer table:
  `supplementary/original_vs_new_pipeline.csv`; plan/state: `revision/LLM_STEP_PLAN.md`.
