# lit_pheno_db
Map peer-reviewed literature for genetic disease and extract data

# DDG2P PubMed → Disease Model Pipeline

A pipeline for mining PubMed for publications relevant to G2P (Gene2Phenotype)
developmental disease records. The pipeline screens abstracts with a fine-tuned
BERT classifier, ranks candidate G2P records with a cross-encoder, and assigns
final mappings with an LLM (DeepSeek-R1-Distill-Qwen-14B). Downstream steps
add HPO phenotype annotations and 2D visualisations of the literature space.

## System requirements

### Hardware
- **GPU is required.** All training and inference steps assume CUDA-capable
  NVIDIA GPUs.
- Reference hardware used for development:
  - BERT classifier inference over PubMed: ~24 h on 1× NVIDIA A100 (80 GB).
  - Cross-encoder inference over candidate pairs: ~24 h on 1× NVIDIA A100.
  - LLM mapping step (DeepSeek-R1-Distill-Qwen-14B via vLLM): ~3 days on
    3× NVIDIA A100.
- ~1–2 TB of disk for downloaded PubMed baseline + intermediate parquet shards.

### Operating system
- Tested on Linux (Ubuntu 22.04, kernel 5.15).
- Not tested on macOS or Windows.

### Software
- Python 3.10 or 3.11
- CUDA 12.x with a compatible NVIDIA driver
- A Java runtime (only required for the `cadmus` full-text fetcher used by the
  optional HPO step; install via `openjdk` in `environment.yml` / the container)
- Python package versions are pinned in `requirements.txt` (and `environment.yml`).
  Versions used during development:
  - torch 2.8, transformers 4.48, datasets 2.20, sentence-transformers 3.0
  - vllm 0.10.1.1
  - pandas 2.2, polars 1.5, pyarrow 17, scikit-learn 1.5
  - pubmed_parser 0.5, lxml 5.2, requests 2.32
  - networkx 3.3, node2vec 0.4.6, owlready2 0.46
  - FastHPOCR 1.0
  - umap-learn 0.5, datamapplot 0.4, matplotlib 3.8
  - (optional, for GPU UMAP/HDBSCAN in visualisation) RAPIDS cuML / cuPy 24.x

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and package
management.

```bash
# 1. Create the environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate

# 2. Install PyTorch matching your CUDA version (see https://pytorch.org)
uv pip install torch==2.8.* --index-url https://download.pytorch.org/whl/cu129

# 3. Install the remaining pinned dependencies
uv pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt`; a conda `environment.yml`
mirroring the same versions is also provided. Run scripts with `uv run`
(e.g. `uv run python annotate_pubmed/crossencode.py --help`) or after activating
the venv.

Typical install time on a normal desktop computer: **15–30 minutes**, dominated
by the PyTorch and vLLM downloads. RAPIDS cuML (optional, used by the GPU
visualisation path) is installed in a **separate** conda environment (Python 3.10,
RAPIDS 24.x) per the official RAPIDS instructions — it is incompatible with the
pinned versions here; `visualisation/ce_tsne.py` falls back to CPU `umap-learn`.

### Containers

For HPC, build the Apptainer/Singularity image (or use the Dockerfile):

```bash
apptainer build litdd.sif containers/litdd.def
apptainer exec --nv litdd.sif python annotate_pubmed/bert_predict_vllm.py --help
```

### Reproducibility / compute

The PubMed-scale run is **single-node, multi-GPU** (vLLM tensor parallelism +
per-shard scripts), not a distributed cluster job. Continuous integration
(`.github/workflows/ci.yml`) runs ruff lint and the CPU unit tests via uv on
every push.

## Demo

**A demo on a "normal" desktop computer is not possible.** Each stage of the
pipeline (BERT inference, cross-encoder inference, and LLM mapping) requires
a high-memory NVIDIA GPU (A100-class) and takes from many hours to several
days end-to-end on the full PubMed baseline. The fine-tuned models and the
DeepSeek-R1-Distill-Qwen-14B LLM together exceed typical consumer GPU memory.

To reproduce results on suitable hardware, see *Instructions for use* below.

## Instructions for use

The pipeline runs in sequence. Working directories for each step are noted.

1. **Download PubMed baseline + daily updates** (`annotate_pubmed/`)
   - `download_pubmed.py` — fetch PubMed XML
   - `pubmed_to_parquet.py` — convert to parquet shards
   - `get_pmids.sh` — collect PMC OA PMIDs

2. **BERT screening of abstracts** (`annotate_pubmed/`)
   - `bert_predict.py` (or the vLLM variant `bert_predict_vllm.py`) using the
     fine-tuned model in `models/precrossencoder_lit_dd_BERT/`.
   - Writes per-shard parquet to `data/bert_processed/`.

3. **Cross-encoder ranking** (`annotate_pubmed/`)
   - `crossencode.py` — scores `(abstract, G2P record)` pairs using
     `models/finetuned_ncbi_medcpt_cross/` and emits the top-5 candidates.

   **How the similarity score is generated.** Each candidate G2P record is
   serialised into a flat string `g2p_id - gene symbol - gene mim - hgnc id -
   previous gene symbols - disease name - disease mim - disease MONDO -
   allelic requirement - cross cutting modifier - confidence - inferred
   variant consequence - variant types - molecular mechanism - molecular
   mechanism categorisation` (`G2P_LGMDE`). For each abstract `tiab` we form
   the pair `(tiab, G2P_LGMDE)` and pass it through the fine-tuned cross-
   encoder (a single transformer that ingests the joined pair via
   `[CLS] tiab [SEP] g2p_lgmde [SEP]` and a 1-logit classification head).
   `model.predict` applies a sigmoid on the logit and returns a relevance
   score in `[0, 1]`. We retain the top-5 highest-scoring G2P records per
   abstract; the score is later thresholded in `final_data_clean.py`
   (default cutoff 0.9).

4. **LLM mapping** (`annotate_pubmed/`)
   - `llm_map.py` — runs `DeepSeek-R1-Distill-Qwen-14B` under vLLM to pick the
     final G2P ID(s) for each abstract from the top-5 candidates.
     Defaults are deterministic (`temperature=0.0`, `top_p=1.0`).

5. **Final clean / dataset assembly** (`annotate_pubmed/`)
   - `final_data_clean.py` filters `(PMID, G2P_ID)` pairs by
     (a) the `top5_cross` score, (b) presence in the G2P CSV (no LLM
     hallucinations), and (c) gene-symbol overlap with PubTator's GNorm2
     annotations resolved through `gene_info.gz`.
   - `--no_gene_check` disables (c) to quantify the gene-mention filter's
     attrition and produce the relaxed corpus (R2-C1/R3.4).
   - `pubtator_genes_api.py` fetches gene annotations per PMID from the
     PubTator3 API and writes them in `gene2pubtator3` format, so the
     gene-mention check can use fresh per-abstract annotations (and avoid the
     bulk file's coverage gaps) — `--gene2pubtator pubtator_api_genes.tsv.gz`.

6. **HPO phenotype annotation** (`hpo_annotations/`)
   - `run_cadmus.py` — fetch full text for the mapped PMIDs via
     [cadmus](https://github.com/biomedicalinformaticsgroup/cadmus). The full
     text itself is **not** redistributed here (publisher permissions); this
     regenerates it from PMIDs.
   - `get_fulltext_df.py` — assemble the cadmus output into a `content_text` parquet.
   - `extract_hpo.py` — annotate with HPO terms via FastHPOCR (build `hp.index`
     from `hp.obo` once with `--build_index`), emitting weighted (frequency-
     preserving) and unweighted HPO profiles per G2P disease.

7. **Visualisation** (`visualisation/`)
   - `ce_tsne.py` — UMAP (cuML where available, else CPU `umap-learn`) + HDBSCAN
     clustering of the cross-encoder embeddings → 2D coords + cluster labels.
   - `datamap_plot.py` — MONDO-labelled `datamapplot` static figure and
     interactive HTML of the literature space.

### Training and evaluation

The training data are in `train_test/annotated_pmid.csv` (columns:
`pmid, g2p_lgmde, label`).

#### Train / test split

`train_test/final_traintest_dataset.py` produces an **80 / 20** group-
stratified split at the PMID-grouping level (so the same PMID/abstract never
appears in both halves) and writes three HuggingFace `save_to_disk` directories:

| Directory          | Purpose                                          |
|--------------------|--------------------------------------------------|
| `ds_bert_train`    | Train portion for the BERT classifier            |
| `ds_cross_train`   | Train portion for the cross-encoder (= `ds_bert_train`) |
| `ds_test`          | Held-out test set — only touched once, after refit |

Inspect the split sizes without writing files:

```bash
python train_test/final_traintest_dataset.py --dry_run
```

**Stricter held-out validation.** `--group_col` selects the leakage-control axis,
including **gene-held-out** (`--group_col gene`) and **disease-held-out**
(`--group_col g2p_id`) splits, where no gene / no G2P disease entry appears in
both train and test. These are stronger generalisation tests than a TIAB-level
split (which can overestimate when the same gene/disease context appears on both
sides). The same CV → refit → held-out-test protocol then runs on the chosen split:

```bash
python train_test/final_traintest_dataset.py --group_col gene    # gene-held-out
python train_test/final_traintest_dataset.py --group_col g2p_id   # disease-held-out
```

#### Methodology — CV-on-train + refit + held-out test

There is no separate validation set. We follow the standard "CV for hyper-
parameter selection, single held-out test" protocol:

  1. **Build the 80/20 split** (above).
  2. **Hyperparameter selection** by 5-fold `StratifiedGroupKFold` cross-
     validation on the *training* set only. The grouping column (`tiab` by
     default) prevents the same abstract from appearing in both the train
     and validation halves of any fold. Each `(learning_rate, weight_decay,
     ...)` combination in the grid is scored by mean fold F1; the best is
     written to a JSON file.
  3. **Refit** on the *full* training set with the selected hyperparameters.
  4. **Evaluate once** on the untouched test set.

The test set is never loaded during steps 2 or 3.

#### Run order

The simplest entry point is the runner at the repo root:

```bash
./run_pipeline.sh --demo    # 100-row sample, tiny CPU models, ~5 min
./run_pipeline.sh --full    # full data, full models, multi-hour on A100
```

Both modes execute the same script sequence (defaults assume you run from
the repo root). The individual commands, in order:

```bash
# 1. 80/20 group-stratified split (writes train_test/{ds_bert_train, ds_cross_train, ds_test})
python train_test/final_traintest_dataset.py --group_col pmid

# 2a. CV hyperparameter search for the BERT classifier (training set only)
python cross_validation/cv_hp_search_bert.py \
    --lr_grid 1e-5 3e-5 \
    --wd_grid 0.1 0.3 \
    --epochs_grid 5
# → writes cross_validation/bert_hp_search.json

# 2b. Refit BERT on the full training set and evaluate once on the test set
python train_test/bert_finetune.py \
    --hp_json cross_validation/bert_hp_search.json

# 3a. Mine hard negatives on the train split (needs the G2P CSV)
python train_test/mine_hard_negatives.py \
    --g2p_csv train_test/G2P_DD_2025-02-15.csv

# 3b. CV hyperparameter search for the cross-encoder
python cross_validation/cv_hp_search_crossencoder.py \
    --g2p_corpus_csv train_test/G2P_DD_2025-02-15.csv

# 3c. Refit cross-encoder on the full hard-negatives train, eval once on ds_test
python train_test/crossencode_finetune.py \
    --hp_json cross_validation/crossencoder_hp_search.json
```

The CV scripts default to a small grid (BERT: `lr × weight_decay` = 4 combos
× 5 folds = 20 trainings, ≈ tens of GPU-hours on 1× A100; cross-encoder:
`lr × epochs` = 2 × 5 = 10 trainings). Override `--lr_grid`, `--wd_grid`,
`--epochs_grid` to widen.

#### Demo

`./run_pipeline.sh --demo` exercises every step end-to-end on 100 rows
sampled from the real annotated dataset, using tiny CPU-friendly substitutes
for each of the three models. See [`demo/README.md`](demo/README.md) for
details and outputs.

#### Baseline benchmarking — `benchmarking/`

Reviewer note: an earlier draft of `run_bert_benchmark.py` evaluated each
baseline by loading the pretrained checkpoint with a freshly-initialised
classification head (`num_labels=2, ignore_mismatched_sizes=True`) and never
fine-tuning. Such a head has never seen any training data and therefore
predicts near-randomly on a 2-class task — that is why the published baseline
F1 ≈ 15% looked anomalously low. The current `run_bert_benchmark.py`
**applies the same CV-on-train + refit + held-out-test protocol to every
baseline** as it does to LitDD-BERT, which is the standard for fair
comparison on a binary classification task.

```bash
# Cheapest: re-use the HP set selected for LitDD-BERT for every baseline
python benchmarking/run_bert_benchmark.py \
    --hp_json cross_validation/bert_hp_search.json \
    --litdd_model_path lit_dd_BERT_best \
    --skip_existing

# Most rigorous: per-baseline CV HP search (slow — multiplies CV cost by number of baselines)
python benchmarking/run_bert_benchmark.py \
    --cv_hp_search \
    --litdd_model_path lit_dd_BERT_best

# Cross-encoder top-K reranker comparison
python benchmarking/run_cross_encoder_benchmark.py \
    --models path/to/finetuned_ncbi_medcpt_cross ncbi/MedCPT-Cross-Encoder
```

### Tests

The `tests/` directory contains a CPU-only test suite (no GPU/torch/vLLM
required): unit tests for the deterministic logic in `llm_map.py`
(prompt building, answer parsing) and `crossencode.py` (top-5 selection, G2P
LGMDE string building), plus an end-to-end test of `final_data_clean.py` on tiny
fixtures. The same suite runs in CI.

```bash
# build the fixtures for the final_data_clean test once
uv run python tests/build_fixtures.py

# run the whole suite
uv run --with numpy --with pandas --with polars --with pyarrow --with pytest pytest tests/ -q
```

### Models

Place / keep fine-tuned weights under `models/`:
- `models/precrossencoder_lit_dd_BERT/` — BERT screening classifier
- `models/finetuned_ncbi_medcpt_cross/` — cross-encoder
- `models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/` — LLM (download from
  Hugging Face)

### Inputs

The pipeline expects a current G2P developmental disease panel CSV
(e.g. `train_test/G2P_DD_2025-02-15.csv`) and a PubMed baseline + updatefiles
download in `annotate_pubmed/data/pubmed_download/`.

