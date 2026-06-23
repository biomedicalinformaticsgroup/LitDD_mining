# Demo

A small CPU-friendly end-to-end run of the LitDD training pipeline. Useful as
a smoke test that every script glues together correctly before kicking off
the full GPU run.

## What the demo does

1. Stratified-samples 100 PMIDs from `train_test/annotated_pmid.csv` (the
   real dataset) into `demo/data/annotated_pmid_demo.csv`.
2. Stages a 134-row G2P CSV (`demo/data/g2p_demo.csv`) covering every G2P_ID
   referenced in those 100 PMIDs plus 50 unrelated extras.
3. Runs the *full training pipeline* on those 100 rows:
   - 80/20 group-stratified split
   - 5-fold (configured to 2 in demo) `StratifiedGroupKFold` CV HP search for
     BERT, with a 1-combo grid
   - Refit BERT on the full demo train, evaluate once on the demo test
   - Hard-negatives mining
   - 5-fold (configured to 2 in demo) HP search for the cross-encoder, with
     a 1-combo grid
   - Refit cross-encoder, evaluate once on the demo test
4. Smoke-tests `annotate_pubmed/final_data_clean.py` on the LLM-output fixture
   in `tests/fixtures/`.

## What the demo skips

PubMed download → BERT inference → cross-encoder inference → LLM mapping is
*not* exercised. Those stages each need >30 GB of GPU memory and the full
PubMed baseline (~1 TB). The demo runs `final_data_clean.py` against a
pre-baked LLM-output parquet fixture instead, which is enough to verify the
cleaner.

## Models used

The demo swaps the production models for tiny CPU-friendly substitutes so
you can run it on a laptop:

| Stage          | Production model                                | Demo model                                    |
|----------------|--------------------------------------------------|-----------------------------------------------|
| BERT classifier | `answerdotai/ModernBERT-large` (395M)            | `distilbert-base-uncased` (66M)               |
| Cross-encoder  | `ncbi/MedCPT-Cross-Encoder` (110M)               | `cross-encoder/ms-marco-MiniLM-L6-v2` (22M)   |
| Embedder       | `abhinand/MedEmbed-large-v0.1` (335M)            | `sentence-transformers/all-MiniLM-L6-v2` (22M)|

## Run

From the repo root:

```bash
./run_pipeline.sh --demo
```

Expected runtime: ~5 minutes on a CPU laptop, ~1 minute on a GPU.

Outputs land under `demo/`:

```
demo/
├── data/
│   ├── annotated_pmid_demo.csv
│   ├── g2p_demo.csv
│   ├── ds_bert_train/
│   ├── ds_cross_train/
│   ├── ds_test/
│   └── hard_negatives_dataset/
├── models/
│   ├── lit_dd_BERT_demo/
│   └── finetuned_cross_demo/
└── results/
    ├── bert_hp.json
    ├── crossencoder_hp.json
    ├── bert_finetune/
    ├── crossencoder_finetune/
    └── final_cleaned_data.csv
```

## Rebuilding demo data

If `train_test/annotated_pmid.csv` changes:

```bash
python demo/build_demo_data.py
```
