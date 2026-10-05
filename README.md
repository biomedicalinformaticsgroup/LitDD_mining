# LitDD

Maps PubMed literature to Gene2Phenotype (G2P) developmental-disorder entries. For every
eligible PubMed record the pipeline decides whether the paper reports a gene-disease
relationship, finds the G2P genes the title and abstract name, and asks a language model
which of those genes' entries the paper is about. The output is a table of (PMID, G2P id)
pairs; the released map is `results/litdd_pubmed2026_final_v6.csv`.

```
PubMed title and abstract shards
  -> screen        litdd.pipeline.bert_predict_vllm     fine-tuned ModernBERT classifier (tmy100000001/LitDD_BERT)
  -> gene gate     litdd.pipeline.gene_candidates       PubTator3 mentions and HGNC names resolved to HGNC ids
  -> shards        litdd.pipeline.build_llm_shards      one candidate list per abstract
  -> adjudication  litdd.pipeline.llm_map               openai/gpt-oss-20b under vLLM, contextualised candidate blocks
  -> clean         litdd.pipeline.final_data_clean      keep ids that exist in the panel and were offered as candidates
```

`supplementary/RUN_FINAL_PIPELINE.md` gives the commands, inputs and evaluation procedure.
`supplementary/DESIGN_NOTES.md` records the reasons behind the design; `CHANGELOG.md` records
dated changes and their measurements.

## Requirements

- Linux, Python 3.10 or 3.11, an NVIDIA GPU with CUDA 12 for the screen and adjudication stages
  (the released run used H100 and A100 nodes). The gene gate, shard builder, clean stage,
  evaluation and tests run on CPU.
- About 1 to 2 TB of disk for the PubMed baseline and its parquet conversion.

## Installation

The GPU stages are built on the vLLM image, which supplies torch, vLLM and transformers:

```bash
docker pull ghcr.io/biomedicalinformaticsgroup/litdd_mining:sha-<commit>   # built by .github/workflows/image.yml
apptainer build litdd.sif containers/litdd.def                          # HPC alternative
```

For a local environment, install vLLM first (it brings the torch and transformers builds it
was released with; the released run used vLLM 0.23.0, torch 2.11 and transformers 5.12), then
the remaining dependencies and the package:

```bash
python -m venv .venv && source .venv/bin/activate
pip install vllm==0.23.*
pip install -r requirements.txt
pip install -e .
```

`environment.yml` builds the same environment with conda. Every stage is run as a module
from the repository root, for example `python -m litdd.pipeline.gene_candidates --help`.

## Repository layout

```
litdd/
├── pipeline/       the corpus pipeline, in stage order: download_pubmed, pubmed_to_parquet,
│                   extract_deleted_pmids, extract_corrections, dedupe_pmids, bert_predict_vllm,
│                   build_bert_positives, gene_candidates, build_llm_shards, llm_map,
│                   final_data_clean; build_context_threads and build_disease_name_aliases
│                   prepare the adjudication stage's inputs
├── genes.py        gene-mention primitives: PubTator loaders, HGNC name matcher, symbol rules
├── gene_resolution.py   HGNC identifier resolution and the MONDO disease lexicon
├── threads.py      the G2P entry label used in the annotated data
├── training/       screen training data and training (litdd/training/README.md)
├── evaluation/     adjudication evaluation, external recall, precision audit
├── experiments/    screen ablations not on the release path
└── viz/            the literature datamap (figure only): embed_papers (MedEmbed bi-encoder),
                    layout_clusters (UMAP and HDBSCAN), datamap_plot (Mondo labels, rendering)
data/               annotation inputs and the released context-thread JSON
results/            the released map
supplementary/      run guide, design notes, result tables
demo/               CPU run of the screen training path
tests/              CPU unit tests (pytest tests/ -q)
containers/         Dockerfile and Apptainer definition
run_pipeline.sh     screen training runner (--demo | --full)
```

## Inputs

| Input | Source |
|---|---|
| G2P developmental-disorder panel and all-panel CSVs | https://www.ebi.ac.uk/gene2phenotype/downloads |
| `gene2pubtator3.gz` | https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/ |
| `hgnc_complete_set.txt` | https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt |
| `Homo_sapiens.gene_info.gz` | https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/ |
| `mondo.obo`, `hp.obo` | http://purl.obolibrary.org/obo/ |
| PubMed baseline and update files | `python -m litdd.pipeline.download_pubmed` |

The annotation used for training and evaluation is `data/annotated_pmid.csv` (columns
`pmid`, `g2p_lgmde`, `label`), with reviewed corrections in `data/annotation_corrections.csv`.

## Models

- `tmy100000001/LitDD_BERT` (branch `main`, release `v2.5`): the screen, a fine-tune of
  `thomas-sounack/BioClinical-ModernBERT-large` (training: `litdd/training/README.md`).
- `openai/gpt-oss-20b` (revision `6cee5e81` in the released run): the adjudication model,
  downloaded by vLLM at run time.

## Results

On the annotated test split (2,731 abstracts, 645 curated; exact set match end to end):
precision 0.865, recall 0.806, F1 0.835. Per-stage confusion matrices:
`supplementary/stage_confusion_matrices.csv`. Configurations evaluated during development:
`supplementary/llm_ablation_ledger.csv` (`supplementary/README_ablation_ledger.md`).

## Training the screen

```bash
./run_pipeline.sh --demo    # 100-abstract sample, small model, CPU
./run_pipeline.sh --full    # full annotated set, released base model, GPU
```

Both run the group-stratified split (`litdd.training.final_traintest_dataset`), a
cross-validated hyperparameter search on the training portion (`cv_hp_search_bert`) and one
refit with a single evaluation on the test portion (`bert_finetune`). The released checkpoint
was trained by `litdd.training.finetune_seeds` on the augmented set described in
`litdd/training/README.md`. `litdd.evaluation.run_bert_benchmark` fine-tunes baseline
encoders with the same protocol for comparison.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

The suite runs on CPU with inline fixtures and covers the gene gate, the disease lexicon,
the shard builder, prompt rendering and answer parsing, the clean stage, the screen's I/O
helpers, the context-thread builder, the evaluator and the training helpers. Continuous
integration (`.github/workflows/ci.yml`) runs ruff and the same suite on every push.

## Licence

MIT (see `LICENSE`).
