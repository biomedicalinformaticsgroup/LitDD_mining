# LitDD — final end-to-end pipeline: where it is and how to run it

**Repo**: `/home/eidf128/eidf128/shared/export/michael/litdd_clean`, branch `cleanup/reviewer-fixes`
(remote `github.com/biomedicalinformaticsgroup/LitDD_mining`).

## The pipeline (as settled 2026-09-17)

    PubMed TIABs
      → 1. screen        litdd/pipeline/bert_predict_vllm.py     HF tmy100000001/LitDD_BERT (main = add20k seed 44)
      → 2. gene gate     litdd/pipeline/gene_candidates.py       --resolution hgnc: every gene resolved by HGNC identifier
                                                                 (PubTator3 GeneID -> HGNC ID -> G2P `hgnc id`), PubTator3
                                                                 TIAB-verified mentions + HGNC names + --symbol_fallback;
                                                                 disease names/abbreviations excluded as gene evidence
                                                                 (derived from HGNC + MONDO; replaces the stop list)
      → 3. adjudication  litdd/pipeline/llm_map.py               openai/gpt-oss-20b, prompts/original_paper_phenotype_v22.txt (default),
                                                                 contextualised threads, same-gene entries from all G2P panels,
                                                                 answers restricted to the DD panel, all candidates, NO score threshold
      → 4. clean         litdd/pipeline/final_data_clean.py      --score_cutoff 0

`prompts/original_paper.txt` is the original decision rubric, kept only to reproduce the submitted
pipeline.

**The cross-encoder is no longer in the cascade** (the gene gate supplies the candidates;
retrieval recall 1.000 on the test set). `tmy100000001/LitDD_crossencoder` is retained on HF for
reproducing the published pipeline and as an optional ranker/audit signal only.

**Measured on the held-out annotated test split** (2,731 abstracts, 645 curated, exact-set match
end to end): **P 0.865 / R 0.806 / F1 0.835** (TP 520, FP 81, FN 125, TN 2,048). Development split
0.813 / 0.825 / 0.819; held-out curated sets end-to-end recall 0.903 raw, 0.911 in scope.
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
  --resolution hgnc --mondo_obo revision/context_build/mondo.obo \
  --mondo_diseases all --context_diseases germline_lineage --context_names label \
  --disease_alias_policy hybrid --symbol_fallback \
  --audit_prefix gate_audit/corpus \
  --out_parquet candidates.parquet
# --audit_prefix writes every HGNC symbol/name considered, why it was kept or excluded, and
# per-rule corpus counts (the Methods numbers). The released 2026 run's audit is
# revision/stoplist_review/hgnc_gate/audit/corpus_hgnc_hybrid_*.

# 3. LLM adjudication  (candidates.parquet needs a top5_cross column: either run
#    crossencode.py --candidates_parquet for scored/ordered candidates, or build the
#    placeholder column as in the direct arm — see litdd/evaluation/build_llm_eval_shards.py)
python litdd/pipeline/llm_map.py --shards_dir shards/ --out_dir out/ \
  --llm_model openai/gpt-oss-20b --temperature 0.0 --top_p 1.0 --seed 0 \
  --reasoning_effort medium --max_model_len 32768 --save_every 5000 \
  --threads context --context_json revision/llm_eval/context_threads_G2P_all_2026-09-10_v21.json \
  --context_drop_fields "Disease Definition,Phenotypes" \
  --panel_siblings_csv revision/G2P_all_2026-09-10.csv \
  --final_panel_csv revision/G2P_DD_2026-06-24.csv

# 4. clean / gate  (--llm_file takes ONE parquet: concatenate the LLM shards first)
python litdd/pipeline/final_data_clean.py --llm_file llm_all.parquet \
  --g2p_file revision/G2P_DD_2026-06-24.csv --gene2pubtator data/reference/gene2pubtator3.gz \
  --gene_info revision/human_gene_info.gz --candidates_parquet candidates.parquet \
  --score_cutoff 0 --output_csv final.csv --no_match_csv nomatch.csv
```

The released 2026 corpus map (`litdd_pubmed2026_final_v6.csv`: 86,513 mappings, 72,939 papers,
2,826 entries) was produced this way. The gate was re-run with `--resolution hgnc`; adjudications
from the previous run were reused only for abstracts whose candidate set was identical, and every
changed or new abstract (3,827) was re-adjudicated: `revision/fullrun_2026/build_v6_shards.py`,
manifests `08_llm_v6_job.yaml` and `09_clean_v6_job.yaml`, driver `run_v6.sh`.

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
any tuning), `external_2026` (held-out curated sets: end-to-end recall 0.903 raw, 0.911 in scope).

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
