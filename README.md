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
- Java 8+ runtime (only required for the `cadmus` PMC full-text fetcher; a
  bundled JRE is included under `cadmus/jre1.8.0_471/`)
- Python package versions are pinned in requirements.txt. 
  Versions used during development:
  - torch 2.4, transformers 4.44, datasets 2.20, sentence-transformers 3.0
  - vllm 0.10.1.1
  - pandas 2.2, polars 1.5, pyarrow 17, scikit-learn 1.5
  - pubmed_parser 0.5, lxml 5.2, requests 2.32
  - networkx 3.3, node2vec 0.4.6, owlready2 0.46
  - FastHPOCR 1.0
  - umap-learn 0.5, datamapplot 0.4, matplotlib 3.8
  - (optional, for GPU UMAP/HDBSCAN in visualisation) RAPIDS cuML / cuPy 24.x

## Installation

```bash
# 1. Create a Python environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch matching your CUDA version (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the remaining dependencies
pip install -r requirements.txt
```

Typical install time on a normal desktop computer: **15–30 minutes**, dominated
by the PyTorch and vLLM downloads. RAPIDS cuML (optional, used by the
visualisation scripts for GPU UMAP) is best installed via conda following the
official RAPIDS instructions.

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

4. **LLM mapping** (`annotate_pubmed/`)
   - `llm_map.py` — runs `DeepSeek-R1-Distill-Qwen-14B` under vLLM to pick the
     final G2P ID(s) for each abstract from the top-5 candidates.

5. **Final clean / dataset assembly** (`annotate_pubmed/`)
   - `final_data_clean.py` and `create_final_dataset_and_plot.ipynb`.

6. **HPO phenotype annotation** (`hpo_annotations/`)
   - `extract_hpo.py` — annotates abstracts/full text with HPO terms via
     FastHPOCR and the included `hp.obo` / `hp.index`.

7. **Visualisation** (`visualisation/`)
   - `mapped_pmid_2d_space.py` / `mapped_pmid_2d_space_v2.py` — UMAP/t-SNE
     embeddings of mapped PMIDs and Mondo layers (uses RAPIDS cuML where
     available, falls back to CPU `umap-learn`).
   - `gene_pathway_nodes_clean.ipynb` — gene/pathway graph.

### Training and evaluation

- **Train/test split and fine-tuning** — `train_test/`
  - `bert_finetune_vals.py`, `crossencode_finetune.py`, `mine_hard_negatives.py`,
    `run_pipeline.py`.
- **5-fold cross-validation** — `cross_validation/`
  - End-to-end driver: `bash run_cv5.sh` (creates folds, trains BERT + cross-
    encoder per fold, runs LLM eval, and aggregates metrics).
- **Benchmarking** — `benchmarking/`
  - `run_bert_benchmark.py`, `run_cross_encoder_benchmark.py`.

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

