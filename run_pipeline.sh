#!/usr/bin/env bash
# End-to-end pipeline runner for LitDD_mining.
#
# Two modes:
#   --demo   100-row stratified sample of the annotated dataset, tiny CPU-friendly
#            models, and a 1-combo × 2-fold CV grid. End-to-end in a few minutes
#            on CPU; sanity-checks every script. Skips PubMed/LLM inference
#            (those require >30 GB GPU and the full PubMed baseline).
#   --full   full annotated dataset, full-size models, full HP grid. Multi-hour
#            on 1× A100 for training + benchmarks; multi-day for end-to-end
#            PubMed inference. The PubMed-inference steps are commented out by
#            default — uncomment after you have downloaded PubMed.
#
# Usage:
#   ./run_pipeline.sh --demo
#   ./run_pipeline.sh --full
#
# All commands assume you run from the repo root and that the Python
# environment in $PYTHON has the dependencies in requirements.txt installed.

set -euo pipefail

MODE=""
PYTHON="${PYTHON:-.venv/bin/python}"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --demo) MODE="demo"; shift ;;
        --full) MODE="full"; shift ;;
        --python) PYTHON="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

[[ -z "$MODE" ]] && usage

cd "$(dirname "$0")"
echo "==> mode: $MODE"
echo "==> python: $PYTHON"
echo "==> repo root: $PWD"

if [[ "$MODE" == "demo" ]]; then
    # ---------------- demo configuration ----------------
    ANNOTATED_CSV="demo/data/annotated_pmid_demo.csv"
    G2P_CSV="demo/data/g2p_demo.csv"
    OUT_DIR="demo/data"
    BERT_MODEL="distilbert-base-uncased"
    CROSS_ENCODER_MODEL="cross-encoder/ms-marco-MiniLM-L6-v2"
    EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    BERT_HP_JSON="demo/results/bert_hp.json"
    CE_HP_JSON="demo/results/crossencoder_hp.json"
    BERT_BEST_DIR="demo/models/lit_dd_BERT_demo"
    CE_BEST_DIR="demo/models/finetuned_cross_demo"
    HARD_NEG_DIR="demo/data/hard_negatives_dataset"
    BERT_OUT_DIR="demo/results/bert_finetune"
    CE_OUT_DIR="demo/results/crossencoder_finetune"
    BERT_GRID=(--lr_grid "1e-5" --wd_grid "0.1" --epochs_grid "1")
    CE_GRID=(--lr_grid "3e-5" --epochs_grid "1")
    N_FOLDS=2
    TRAIN_BS=8
    EVAL_BS=8
    REFIT_EPOCHS=1
elif [[ "$MODE" == "full" ]]; then
    # ---------------- full configuration ----------------
    ANNOTATED_CSV="train_test/annotated_pmid.csv"
    G2P_CSV="${G2P_CSV:-train_test/G2P_DD_2026-06-24.csv}"  # user-supplied, not in repo
    OUT_DIR="train_test"
    BERT_MODEL="${BERT_MODEL:-thomas-sounack/BioClinical-ModernBERT-large}"  # screen fine-tune base
    CROSS_ENCODER_MODEL="ncbi/MedCPT-Cross-Encoder"
    EMBED_MODEL="abhinand/MedEmbed-large-v0.1"
    BERT_HP_JSON="cross_validation/bert_hp_search.json"
    CE_HP_JSON="cross_validation/crossencoder_hp_search.json"
    BERT_BEST_DIR="train_test/lit_dd_BERT_best"
    CE_BEST_DIR="train_test/finetuned_ncbi_medcpt_cross"
    HARD_NEG_DIR="train_test/hard_negatives_dataset"
    BERT_OUT_DIR="train_test/bert_finetune_results"
    CE_OUT_DIR="train_test/finetuned_cross_encoders"
    BERT_GRID=(--lr_grid "1e-5" "3e-5" --wd_grid "0.1" "0.3" --epochs_grid "5")
    CE_GRID=(--lr_grid "1e-5" "3e-5" --epochs_grid "2")
    N_FOLDS=5
    TRAIN_BS=32
    EVAL_BS=32
    REFIT_EPOCHS=5
fi

run() { echo; echo "==> $*"; "$@"; }

# 1. Build the train/test split
run $PYTHON train_test/final_traintest_dataset.py \
    --annotated_csv "$ANNOTATED_CSV" \
    --out_dir "$OUT_DIR" \
    --group_col pmid

# 2a. CV HP search for BERT (training set only)
run $PYTHON cross_validation/cv_hp_search_bert.py \
    --train_ds_dir "$OUT_DIR/ds_bert_train" \
    --input_model "$BERT_MODEL" \
    --group_col tiab \
    --n_folds "$N_FOLDS" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" \
    --out_json "$BERT_HP_JSON" \
    "${BERT_GRID[@]}"

# 2b. Refit BERT on the full training set, evaluate once on the test set
run $PYTHON train_test/bert_finetune.py \
    --train_ds_dir "$OUT_DIR/ds_bert_train" \
    --test_ds_dir "$OUT_DIR/ds_test" \
    --input_model "$BERT_MODEL" \
    --hp_json "$BERT_HP_JSON" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" --epochs "$REFIT_EPOCHS" \
    --output_dir "$BERT_OUT_DIR" \
    --best_model_dir "$BERT_BEST_DIR"

# 3a. Mine hard negatives on the train split
run $PYTHON train_test/mine_hard_negatives.py \
    --ds_cross_dir "$OUT_DIR/ds_cross_train" \
    --g2p_csv "$G2P_CSV" \
    --out_dir "$HARD_NEG_DIR" \
    --embed_model "$EMBED_MODEL"

# 3b. CV HP search for the cross-encoder (training set only, hard-negs mined per fold)
run $PYTHON cross_validation/cv_hp_search_crossencoder.py \
    --train_ds_dir "$OUT_DIR/ds_cross_train" \
    --g2p_corpus_csv "$G2P_CSV" \
    --input_model "$CROSS_ENCODER_MODEL" \
    --embed_model "$EMBED_MODEL" \
    --group_col tiab \
    --n_folds "$N_FOLDS" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" \
    --out_json "$CE_HP_JSON" \
    "${CE_GRID[@]}"

# 3c. Refit cross-encoder, evaluate once on the test set
run $PYTHON train_test/crossencode_finetune.py \
    --data_dir "$OUT_DIR" \
    --hard_negatives_subdir "$(basename "$HARD_NEG_DIR")" \
    --test_subdir "ds_test" \
    --input_model "$CROSS_ENCODER_MODEL" \
    --hp_json "$CE_HP_JSON" \
    --output_dir "$CE_OUT_DIR" \
    --best_model_dir "$CE_BEST_DIR" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS"

# 4. Smoke-test the cleaning step on the LLM-output fixture (demo) or real shards (full).
if [[ "$MODE" == "demo" ]]; then
    run $PYTHON annotate_pubmed/final_data_clean.py \
        --llm_file tests/fixtures/llm_shard_sample.parquet \
        --g2p_file tests/fixtures/g2p_sample.csv \
        --gene2pubtator tests/fixtures/gene2pubtator_sample.tsv.gz \
        --gene_info tests/fixtures/gene_info_sample.gz \
        --score_cutoff 0.5 \
        --output_csv demo/results/final_cleaned_data.csv

    echo
    echo "==> Demo complete. Outputs:"
    echo "    BERT model        : $BERT_BEST_DIR"
    echo "    Cross-encoder     : $CE_BEST_DIR"
    echo "    Bert HP JSON      : $BERT_HP_JSON"
    echo "    Cross HP JSON     : $CE_HP_JSON"
    echo "    Final clean CSV   : demo/results/final_cleaned_data.csv"
else
    cat <<'EOF'

==> Full training/evaluation pipeline complete.

Next steps for end-to-end PubMed inference (multi-day on multi-A100; uncomment in this script when you are ready to run):

    # PUBMED_DIR=data/pubmed_download_2026   # MUST be a fresh dir for a new baseline year:
    #                                        # NCBI reissues the baseline each December under a
    #                                        # new pubmedYYn* prefix, and mixing years in one
    #                                        # directory duplicates every record downstream.
    # python annotate_pubmed/download_pubmed.py --download_dir "$PUBMED_DIR" --workers 4
    # python annotate_pubmed/pubmed_to_parquet.py --download_dir "$PUBMED_DIR" --workers 16
    # python annotate_pubmed/dedupe_pmids.py --download_dir "$PUBMED_DIR"   # DeleteCitation + reissues
    # python annotate_pubmed/bert_predict_vllm.py --model "$BERT_BEST_DIR" \
    #     --input_dir "$PUBMED_DIR/parquet_download_files" --shard $i --num_shards 8
    # python annotate_pubmed/build_bert_positives.py
    # # --num_shards 32 so 8 workers all get work and preemption costs <=1/32 of the stage
    # python annotate_pubmed/crossencode.py --g2p_csv "$G2P_CSV" --shard $i --num_shards 32
    # python annotate_pubmed/llm_map.py --shards_dir crossencoded_shards \
    #     --llm_model openai/gpt-oss-20b --shard_index $i --num_shards 8
    # python annotate_pubmed/final_data_clean.py \
    #     --llm_file <pubmed_..._llm.parquet> \
    #     --g2p_file "$G2P_CSV" \
    #     --gene2pubtator path/to/gene2pubtator3.gz \
    #     --gene_info path/to/gene_info.gz \
    #     --score_cutoff 0.9 \
    #     --output_csv final_cleaned_data.csv

You can also run the fair-comparison baseline benchmark:

    python benchmarking/run_bert_benchmark.py \
        --hp_json cross_validation/bert_hp_search.json \
        --litdd_model_path train_test/lit_dd_BERT_best --skip_existing
EOF
fi
