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
(e.g. `uv run python litdd/pipeline/crossencode.py --help`) or after activating
the venv.

Typical install time on a normal desktop computer: **15–30 minutes**, dominated
by the PyTorch and vLLM downloads. RAPIDS cuML (optional, used by the GPU
visualisation path) is installed in a **separate** conda environment (Python 3.10,
RAPIDS 24.x) per the official RAPIDS instructions — it is incompatible with the
pinned versions here; `litdd/viz/ce_tsne.py` falls back to CPU `umap-learn`.

### Containers

For HPC, build the Apptainer/Singularity image (or use the Dockerfile):

```bash
apptainer build litdd.sif containers/litdd.def
apptainer exec --nv litdd.sif python litdd/pipeline/bert_predict_vllm.py --help
```

### Reproducibility / compute

The PubMed-scale run is **single-node, multi-GPU** (vLLM tensor parallelism +
per-shard scripts), not a distributed cluster job. Continuous integration
(`.github/workflows/ci.yml`) runs ruff lint and the CPU unit tests via uv on
every push.

## Repository layout

```
litdd/                  importable package  (pip install -e .)
├── pipeline/           the deployment cascade, in stage order:
│                       download_pubmed → pubmed_to_parquet → dedupe_pmids →
│                       bert_predict[_vllm] → build_bert_positives →
│                       gene_candidates → crossencode → llm_map →
│                       final_data_clean
├── training/           canonical model training + CV hyperparameter search
│                       (see litdd/training/README.md for which script produced
│                        which released model)
├── evaluation/         benchmarks, external-recall harness, precision audit
├── hpo/                downstream HPO extraction from full text
└── viz/                figures
experiments/            revision-era ablations — NOT on the release path
data/                   annotation inputs; training datasets are written here
results/                published output maps
demo/                   CPU smoke test (`./run_pipeline.sh --demo`)
tests/                  CPU-only unit tests
containers/             Apptainer definition + Dockerfile
run_pipeline.sh         end-to-end runner (`--demo` | `--full`)
```

Every stage is a standalone CLI with `--help`. The package is importable so tests and
cross-stage imports work without `sys.path` manipulation:

```bash
pip install -e .
```

## Demo

**A demo on a "normal" desktop computer is not possible.** Each stage of the
pipeline (BERT inference, cross-encoder inference, and LLM mapping) requires
a high-memory NVIDIA GPU (A100-class) and takes from many hours to several
days end-to-end on the full PubMed baseline. The fine-tuned models and the
DeepSeek-R1-Distill-Qwen-14B LLM together exceed typical consumer GPU memory.

To reproduce results on suitable hardware, see *Instructions for use* below.

## Instructions for use

The pipeline runs in sequence. Working directories for each step are noted.

1. **Download PubMed baseline + daily updates** (`litdd/pipeline/`)
   - `download_pubmed.py` — fetch PubMed XML
   - `pubmed_to_parquet.py` — convert to parquet shards
   - `get_pmids.sh` — collect PMC OA PMIDs

2. **BERT screening of abstracts** (`litdd/pipeline/`)
   - `bert_predict.py` (or the vLLM variant `bert_predict_vllm.py`) using
     [`tmy100000001/LitDD_BERT`](https://huggingface.co/tmy100000001/LitDD_BERT),
     a fine-tune of `thomas-sounack/BioClinical-ModernBERT-large`. Override with
     `--model_path` / `$LITDD_BERT_MODEL` for a local checkpoint.
   - Only English records published after 1980 are classified.
   - **No truncation in practice.** The screen is ModernBERT, so `--max_length`
     defaults to its full **8,192-token** context. The earlier 512-token cap was a
     relic of the BERT-large base this model replaced; it truncated ~1% of
     abstracts (observed maximum ~800 tokens). ModernBERT unpads, so the larger
     window costs no throughput on short sequences.
   - Writes per-shard parquet to `data/bert_processed/`.
   - **Requires `transformers >= 4.48`** — ModernBERT support landed in that
     release; earlier versions fail with `KeyError: 'modernbert'`.

3. **Gene-candidate selection** (`litdd/pipeline/`)
   - `gene_candidates.py` — restricts each abstract to the G2P entries whose
     gene it actually mentions, using PubTator3's `gene2pubtator3` bulk
     annotations (human-filtered) plus an HGNC descriptive-name dictionary that
     catches papers naming the gene product but not the symbol.
   - This is a **gate**, so its recall bounds the pipeline. Measured before
     adoption: it retains **98.8%** of true (paper, gene) pairs on the
     independent curated sets, **99.3%** with the name complement
     (`litdd/evaluation/gene_filter_recall.py`).
   - It collapses the candidate set from all 2,861 G2P entries to ~2 on average,
     removing ~1,300× of the cross-encoder's work and making the number of
     candidates shown to the LLM data-driven rather than a fixed top-5.

4. **Cross-encoder ranking** (`litdd/pipeline/`)
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

5. **LLM mapping** (`litdd/pipeline/`)
   - `llm_map.py` — runs the adjudication LLM under vLLM to pick the final
     G2P ID(s) for each abstract from its candidates. Defaults are deterministic
     (`temperature=0.0`, `top_p=1.0`) so the deployed map is reproducible from
     the same inputs.
   - The number of candidates is **data-driven** — whatever the gene gate
     selected — rather than a fixed five; the prompt states the actual count and
     numbers the candidates. `--max_candidates 5` reproduces the original
     fixed-top-5 behaviour for comparison.
   - Work is split across workers by **row**, so any number of workers can share
     any number of shard files, and the stage **resumes** from its checkpoint
     rather than restarting a shard from the beginning.

6. **Final clean / dataset assembly** (`litdd/pipeline/`)
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

7. **HPO phenotype annotation** (`litdd/hpo/`)
   - `run_cadmus.py` — fetch full text for the mapped PMIDs via
     [cadmus](https://github.com/biomedicalinformaticsgroup/cadmus). The full
     text itself is **not** redistributed here (publisher permissions); this
     regenerates it from PMIDs.
   - `get_fulltext_df.py` — assemble the cadmus output into a `content_text` parquet.
   - `extract_hpo.py` — annotate with HPO terms via FastHPOCR (build `hp.index`
     from `hp.obo` once with `--build_index`), emitting weighted (frequency-
     preserving) and unweighted HPO profiles per G2P disease.

8. **Visualisation** (`litdd/viz/`)
   - `ce_tsne.py` — UMAP (cuML where available, else CPU `umap-learn`) + HDBSCAN
     clustering of the cross-encoder embeddings → 2D coords + cluster labels.
   - `datamap_plot.py` — MONDO-labelled `datamapplot` static figure and
     interactive HTML of the literature space.

### Training and evaluation

The training data are in `data/annotated_pmid.csv` (columns:
`pmid, g2p_lgmde, label`).

#### Train / test split

`litdd/training/final_traintest_dataset.py` produces an **80 / 20** group-
stratified split at the PMID-grouping level (so the same PMID/abstract never
appears in both halves) and writes three HuggingFace `save_to_disk` directories:

| Directory          | Purpose                                          |
|--------------------|--------------------------------------------------|
| `ds_bert_train`    | Train portion for the BERT classifier            |
| `ds_cross_train`   | Train portion for the cross-encoder (= `ds_bert_train`) |
| `ds_test`          | Held-out test set — only touched once, after refit |

Inspect the split sizes without writing files:

```bash
python litdd/training/final_traintest_dataset.py --dry_run
```

**Stricter held-out validation.** `--group_col` selects the leakage-control axis,
including **gene-held-out** (`--group_col gene`) and **disease-held-out**
(`--group_col g2p_id`) splits, where no gene / no G2P disease entry appears in
both train and test. These are stronger generalisation tests than a TIAB-level
split (which can overestimate when the same gene/disease context appears on both
sides). The same CV → refit → held-out-test protocol then runs on the chosen split:

```bash
python litdd/training/final_traintest_dataset.py --group_col gene    # gene-held-out
python litdd/training/final_traintest_dataset.py --group_col g2p_id   # disease-held-out
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
# 1. 80/20 group-stratified split (writes litdd/training/{ds_bert_train, ds_cross_train, ds_test})
python litdd/training/final_traintest_dataset.py --group_col pmid

# 2a. CV hyperparameter search for the BERT classifier (training set only)
python litdd/training/cv_hp_search_bert.py \
    --lr_grid 1e-5 3e-5 \
    --wd_grid 0.1 0.3 \
    --epochs_grid 5
# → writes litdd/training/bert_hp_search.json

# 2b. Refit BERT on the full training set and evaluate once on the test set
python litdd/training/bert_finetune.py \
    --hp_json litdd/training/bert_hp_search.json

# 3a. Mine hard negatives on the train split (needs the G2P CSV)
python litdd/training/mine_hard_negatives.py \
    --g2p_csv litdd/training/G2P_DD_2025-02-15.csv

# 3b. CV hyperparameter search for the cross-encoder
python litdd/training/cv_hp_search_crossencoder.py \
    --g2p_corpus_csv litdd/training/G2P_DD_2025-02-15.csv

# 3c. Refit cross-encoder on the full hard-negatives train, eval once on ds_test
python litdd/training/crossencode_finetune.py \
    --hp_json litdd/training/crossencoder_hp_search.json
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

#### Baseline benchmarking — `litdd/evaluation/`

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
python litdd/evaluation/run_bert_benchmark.py \
    --hp_json litdd/training/bert_hp_search.json \
    --litdd_model_path lit_dd_BERT_best \
    --skip_existing

# Most rigorous: per-baseline CV HP search (slow — multiplies CV cost by number of baselines)
python litdd/evaluation/run_bert_benchmark.py \
    --cv_hp_search \
    --litdd_model_path lit_dd_BERT_best

# Cross-encoder top-K reranker comparison
python litdd/evaluation/run_cross_encoder_benchmark.py \
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

Both LitDD models are published on Hugging Face and are the defaults — no manual
download needed:

- [`tmy100000001/LitDD_BERT`](https://huggingface.co/tmy100000001/LitDD_BERT) —
  screening classifier (`--model_path` / `$LITDD_BERT_MODEL`)
- [`tmy100000001/LitDD_crossencoder`](https://huggingface.co/tmy100000001/LitDD_crossencoder) —
  cross-encoder (`--model_path` / `$LITDD_CROSSENCODER`)
- the adjudication LLM is downloaded from Hugging Face by vLLM at run time

To run fully offline, pre-download them and pass local paths to the same flags.

### Inputs

The pipeline expects a current G2P developmental disease panel CSV
(e.g. `litdd/training/G2P_DD_2026-06-24.csv`) and a PubMed baseline + updatefiles
download in `litdd/pipeline/data/pubmed_download/`.

