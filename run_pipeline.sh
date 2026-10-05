#!/usr/bin/env bash
# Training runner for the LitDD screen.
#
# Modes:
#   --demo   100-row sample of the annotated dataset, a small CPU model and a 1-combination
#            2-fold CV grid; runs in a few minutes on CPU.
#   --full   full annotated dataset, the released base model and the full grid; hours on a GPU.
#
# The corpus pipeline (screen, gene gate, shards, adjudication, clean) is documented in
# supplementary/RUN_FINAL_PIPELINE.md and needs a GPU node.
#
# Usage:
#   ./run_pipeline.sh --demo
#   ./run_pipeline.sh --full [--python PATH]

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

if [[ "$MODE" == "demo" ]]; then
    ANNOTATED_CSV="demo/data/annotated_pmid_demo.csv"
    OUT_DIR="demo/data"
    BERT_MODEL="distilbert-base-uncased"
    BERT_HP_JSON="demo/results/bert_hp.json"
    BERT_BEST_DIR="demo/models/lit_dd_BERT_demo"
    BERT_OUT_DIR="demo/results/bert_finetune"
    BERT_GRID=(--lr_grid "1e-5" --wd_grid "0.1" --epochs_grid "1")
    N_FOLDS=2
    TRAIN_BS=8
    EVAL_BS=8
    REFIT_EPOCHS=1
else
    ANNOTATED_CSV="data/annotated_pmid.csv"
    OUT_DIR="data"
    BERT_MODEL="${BERT_MODEL:-thomas-sounack/BioClinical-ModernBERT-large}"
    BERT_HP_JSON="litdd/training/bert_hp_search.json"
    BERT_BEST_DIR="litdd/training/lit_dd_BERT_best"
    BERT_OUT_DIR="litdd/training/bert_finetune_results"
    BERT_GRID=(--lr_grid "1e-5" "3e-5" --wd_grid "0.1" "0.3" --epochs_grid "5")
    N_FOLDS=5
    TRAIN_BS=32
    EVAL_BS=32
    REFIT_EPOCHS=5
fi

run() { echo; echo "==> $*"; "$@"; }

# 1. Group-stratified train/test split
run $PYTHON -m litdd.training.final_traintest_dataset \
    --annotated_csv "$ANNOTATED_CSV" \
    --out_dir "$OUT_DIR" \
    --group_col pmid

# 2. Cross-validated hyperparameter search on the training portion
run $PYTHON -m litdd.training.cv_hp_search_bert \
    --train_ds_dir "$OUT_DIR/ds_bert_train" \
    --input_model "$BERT_MODEL" \
    --group_col tiab \
    --n_folds "$N_FOLDS" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" \
    --out_json "$BERT_HP_JSON" \
    "${BERT_GRID[@]}"

# 3. Refit on the full training portion, evaluate once on the test portion
run $PYTHON -m litdd.training.bert_finetune \
    --train_ds_dir "$OUT_DIR/ds_bert_train" \
    --test_ds_dir "$OUT_DIR/ds_test" \
    --input_model "$BERT_MODEL" \
    --hp_json "$BERT_HP_JSON" \
    --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" --epochs "$REFIT_EPOCHS" \
    --output_dir "$BERT_OUT_DIR" \
    --best_model_dir "$BERT_BEST_DIR"

echo
echo "==> Done. Outputs:"
echo "    screen model      : $BERT_BEST_DIR"
echo "    hyperparameters   : $BERT_HP_JSON"
if [[ "$MODE" == "full" ]]; then
    cat <<'EOF'

The released screen was trained with litdd.training.finetune_seeds on the augmented
annotation set (see litdd/training/README.md). To run the corpus pipeline, follow
supplementary/RUN_FINAL_PIPELINE.md. Unit tests: pytest tests/ -q
EOF
fi
