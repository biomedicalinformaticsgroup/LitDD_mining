# Demo

A small CPU run of the screen training path, as a check that the scripts fit together
before a full GPU run.

## What the demo does

1. Uses a 100-abstract stratified sample of the annotated dataset
   (`demo/data/annotated_pmid_demo.csv`) and a 134-row G2P CSV (`demo/data/g2p_demo.csv`).
2. Builds the group-stratified train/test split.
3. Runs a 2-fold cross-validated hyperparameter search for the screen with a one-combination
   grid, on a small model (`distilbert-base-uncased`).
4. Refits the screen on the demo training portion and evaluates once on the demo test portion.

The corpus stages (PubMed download, screen inference, gene gate, adjudication, clean) are not
exercised; they need a GPU and the PubMed baseline. Their deterministic logic is covered by
`pytest tests/ -q`, which runs on CPU with inline fixtures.

## Run

From the repository root:

```bash
./run_pipeline.sh --demo
```

Outputs land under `demo/`:

```
demo/
├── data/
│   ├── annotated_pmid_demo.csv
│   ├── g2p_demo.csv
│   ├── ds_bert_train/
│   └── ds_test/
├── models/
│   └── lit_dd_BERT_demo/
└── results/
    ├── bert_hp.json
    └── bert_finetune/
```

## Rebuilding the demo data

```bash
python -m demo.build_demo_data --g2p_csv G2P_DD.csv
```
