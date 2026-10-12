# LitDD

Code and data for *PubMed-wide Classification and Disease-level Mapping of Peer-reviewed
Literature for Genetic Developmental Disorders in Gene2Phenotype* (Yates et al.).

LitDD maps PubMed literature to Gene2Phenotype (G2P) developmental-disorder (DDG2P) entries. For
every eligible PubMed record the pipeline decides whether the paper reports a gene-disease
relationship, finds the DDG2P genes the title and abstract name, and asks a language model which
of those genes' entries the paper is about. The output is a table of (PMID, G2P id) pairs; the
released map is `results/litdd_pubmed2026_final_v6.csv` (86,513 pairs, one row per pair, over
72,939 papers and 2,826 entries).

```
PubMed title and abstract shards
  -> 1. screen        litdd.pipeline.bert_predict_vllm   fine-tuned ModernBERT classifier (tmy100000001/LitDD_BERT)
  -> 2. gene gate     litdd.pipeline.gene_candidates     PubTator3 mentions, HGNC names and symbols resolved to HGNC ids
  -> 3. shards        litdd.pipeline.build_llm_shards    one candidate list per abstract
  -> 4. adjudication  litdd.pipeline.llm_map             openai/gpt-oss-20b under vLLM, contextualised candidate blocks
  -> 5. clean         litdd.pipeline.final_data_clean    keep ids that are in the panel and were offered as candidates
```

## Requirements

- Linux (tested on Ubuntu 22.04), Python 3.10 or 3.11.
- An NVIDIA GPU with CUDA 12 for the screen and adjudication stages (the released run used
  H100 80 GB and A100 nodes). Ingestion, the gene gate, the shard builder, the clean stage,
  evaluation and the demo run on CPU.
- About 1 to 2 TB of disk for the PubMed baseline and its parquet conversion.
- Versions of the released run: vLLM 0.23.0, PyTorch 2.11, transformers 5.12 (the container
  image below); minimum versions of the other dependencies are in `requirements.txt`.

## Installation

The GPU stages are built on the vLLM image, which supplies torch, vLLM and transformers:

```bash
docker pull ghcr.io/biomedicalinformaticsgroup/litdd_mining:sha-<commit>   # built by .github/workflows/image.yml
apptainer build litdd.sif containers/litdd.def                          # HPC alternative
```

For a local environment, install vLLM first (it brings the torch and transformers builds it
was released with), then the remaining dependencies and the package:

```bash
python -m venv .venv && source .venv/bin/activate
pip install vllm==0.23.*
pip install -r requirements.txt
pip install -e .
```

`environment.yml` builds an equivalent environment with conda. For the CPU demo only, install
CPU torch and transformers instead of vLLM (`pip install torch --index-url
https://download.pytorch.org/whl/cpu "transformers>=4.48" -r requirements.txt && pip install -e .`);
this took about 3.5 minutes on a 16-core Linux machine with an empty pip cache. Installing vLLM or
pulling the container image takes longer and depends on the network.

Every stage is run as a module from the repository root and prints its options with `--help`,
for example `python -m litdd.pipeline.gene_candidates --help`.

## Demo

```bash
./run_pipeline.sh --demo
```

The demo runs the screen-training path on CPU: a group-stratified train/test split of a
100-PMID (102-row) sample of the annotated set (`demo/data/annotated_pmid_demo.csv`, title text
only), a 2-fold cross-validated hyperparameter search with one grid point, and a refit of a small
model (`distilbert-base-uncased`, downloaded on first use) evaluated once on the test portion. It
took about 75 seconds on a 16-core CPU, plus a one-off download of the model (about 270 MB). Expected output: `demo/results/bert_hp.json` (the selected
hyperparameters and fold F1), `demo/results/bert_finetune/` (test metrics) and the model in
`demo/models/lit_dd_BERT_demo/`; the console prints `[TEST metrics]` and ends with `==> Done. Outputs:`.
On the 20 test rows our run gave accuracy 0.90 and F1 0.80; values can differ slightly between
machines, and the cross-validation folds of 41 rows are too small to be informative (fold F1 can be 0).
See `demo/README.md`.

## Repository layout

```
litdd/
├── pipeline/       the corpus pipeline, in stage order: download_pubmed, extract_deleted_pmids,
│                   extract_corrections, pubmed_to_parquet, dedupe_pmids, bert_predict_vllm,
│                   build_bert_positives, gene_candidates, build_llm_shards, llm_map,
│                   final_data_clean; build_context_threads and build_disease_name_aliases
│                   prepare the adjudication stage's inputs
├── genes.py        gene-mention primitives: PubTator loaders, HGNC name matcher, symbol rules
├── gene_resolution.py   HGNC identifier resolution and the MONDO disease lexicon
├── threads.py      the G2P entry label used in the annotated data
├── training/       screen training data and training (litdd/training/README.md)
├── evaluation/     adjudication evaluation, external recall, precision audit, comparisons
├── experiments/    screen ablations not on the release path
└── viz/            the literature datamap (figure only): embed_papers (MedEmbed bi-encoder),
                    layout_clusters (UMAP and HDBSCAN), datamap_plot (Mondo labels, rendering)
data/               annotation (PMIDs and labels), its corrections, the released context threads,
                    supplementary_data/ (Supplementary Data 1-7) and source_data/ (Source Data)
results/            the released map
demo/               CPU run of the screen training path
containers/         Dockerfile and Apptainer definition
run_pipeline.sh     screen training runner (--demo | --full)
```

## Data

- `data/annotated_pmid.csv`: the annotated set (`pmid`, `g2p_lgmde`, `label`) with the reviewed
  corrections in `data/annotation_corrections.csv` already applied. It carries no title or
  abstract text; `litdd.training.attach_tiab` adds it (see Training the screen).
- `data/supplementary_data/`: Supplementary Data 1-7 of the paper (Supplementary Data 1 without
  abstract text; the journal's copy includes it), described in its `README.md`.
- `data/source_data/`: the Source Data of the paper's figures and tables.
- `results/litdd_pubmed2026_final_v6.csv`: the released map.

## Models

- `tmy100000001/LitDD_BERT` (revision `v2.5`, the same weights as branch `main`): the screen, a
  fine-tune of `thomas-sounack/BioClinical-ModernBERT-large`. `bert_predict_vllm --revision`
  defaults to `v2.5`.
- `openai/gpt-oss-20b` (revision `6cee5e81`): the adjudication model. `llm_map --llm_revision`
  defaults to this revision. Both are downloaded from Hugging Face at run time; on compute nodes
  without internet access, stage them into a Hugging Face cache first and point `HF_HOME` at it.

## Running the LitDD pipeline

### Inputs

| Input | Source |
|---|---|
| G2P developmental-disorder panel CSV (`G2P_DD.csv`) | https://www.ebi.ac.uk/gene2phenotype/downloads (the release used is Supplementary Data 3) |
| G2P all-panel CSV (`G2P_all.csv`) | same page, every panel; used for the context threads and the adjudication stage |
| `gene2pubtator3.gz` | https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/ |
| `hgnc_complete_set.txt` | https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt |
| `gene_info.gz` (Homo sapiens) | https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz |
| `mondo.obo`, `hp.obo` (`mondo.owl` for the datamap) | http://purl.obolibrary.org/obo/ |
| PubMed baseline and update files | step 0 below |

### Preparatory step: context threads

The adjudication stage shows each candidate G2P entry as a multi-line block built from the
all-panel export, MONDO and HPO. The released block set is
`data/context_threads_G2P_all_2026-09-10.json`. To rebuild it for a new export:

```bash
python -m litdd.pipeline.build_context_threads --g2p_csv G2P_all.csv \
  --mondo_obo mondo.obo --hp_obo hp.obo --out_json context_threads.json
```

The builder removes disease-name aliases from each entry's former gene symbols using
`litdd/pipeline/data/disease_name_aliases.tsv`, which `litdd.pipeline.build_disease_name_aliases`
regenerates from HGNC and MONDO. Build the context threads from the same all-panel export that
step 4 passes as `--panel_siblings_csv`.

### Run on a corpus (GPU for steps 1 and 4)

Container: `ghcr.io/biomedicalinformaticsgroup/litdd_mining` (built by `.github/workflows/image.yml`
from `containers/Dockerfile`; vLLM 0.23.0), tagged by commit SHA.

```bash
# 0. corpus: download, list withdrawn and corrected records, convert to parquet, de-duplicate
#    (keeps the latest version of each PMID; removes withdrawn records, retraction, erratum and
#    expression-of-concern notices, duplicate publications, and retracted, superseded or
#    concern-flagged papers)
python -m litdd.pipeline.download_pubmed --download_dir dl/
python -m litdd.pipeline.extract_deleted_pmids --raw_dir dl/raw_download_files --out dl/deleted_pmids.txt
python -m litdd.pipeline.extract_corrections --raw_dir dl/raw_download_files --out dl/corrections.csv
python -m litdd.pipeline.pubmed_to_parquet --download_dir dl/ --workers 16
python -m litdd.pipeline.dedupe_pmids --download_dir dl/            # writes dl/pmid_keep.parquet

# 1. screen (English records published after 1980; about 0.5% of records are positive)
python -m litdd.pipeline.bert_predict_vllm --model tmy100000001/LitDD_BERT \
  --input_dir dl/parquet_download_files --processed_dir screened/ --keep_parquet dl/pmid_keep.parquet
python -m litdd.pipeline.build_bert_positives --processed_dir screened/ --out_path screened/bert_positive.parquet

# 2. gene gate (DDG2P genes; HGNC-identifier resolution; disease names excluded).
#    --audit_prefix writes every symbol and name considered, why it was kept or excluded, and per-rule counts.
python -m litdd.pipeline.gene_candidates --input_parquet screened/bert_positive.parquet \
  --g2p_csv G2P_DD.csv --gene2pubtator gene2pubtator3.gz --hgnc hgnc_complete_set.txt \
  --mondo_obo mondo.obo --audit_prefix gate_audit/corpus --out_parquet candidates.parquet

# 3. adjudication shards (--num_shards or --rows_per_shard)
python -m litdd.pipeline.build_llm_shards --candidates_parquet candidates.parquet --out_dir shards/ --num_shards 8

# 4. adjudication. The defaults are the released settings (gpt-oss-20b revision 6cee5e81, temperature 0,
#    top_p 1, seed 0, reasoning effort medium, 8,192-token budget, max_model_len 32768, Disease Definition
#    and Phenotypes dropped from the candidate blocks). Add --shard_index i --num_shards N per GPU worker.
python -m litdd.pipeline.llm_map --shards_dir shards/ --out_dir out/ --llm_model openai/gpt-oss-20b \
  --context_json data/context_threads_G2P_all_2026-09-10.json \
  --panel_siblings_csv G2P_all.csv --final_panel_csv G2P_DD.csv --save_every 5000
python -c "import polars as pl; pl.scan_parquet('out/*__llm.parquet').sink_parquet('llm_all.parquet')"

# 5. clean: keep ids that are in G2P_DD and were offered as candidates for that abstract
python -m litdd.pipeline.final_data_clean --llm_file llm_all.parquet --g2p_file G2P_DD.csv \
  --candidates_parquet candidates.parquet --gene2pubtator gene2pubtator3.gz --gene_info gene_info.gz \
  --output_csv final.csv --no_match_csv nomatch.csv
```

`final.csv` has one row per (PMID, G2P id) pair, the format of the released map. `llm_map` writes
a `run_meta.json` per shard with the settings, software versions, GPU and throughput.

### Evaluate a run on the annotated test set

The annotation with title and abstract text (`anno.csv`: `pmid`, `g2p_id`, `tiab`, `label`) is
Supplementary Data 1 joined to its text (`python -m litdd.training.attach_tiab --annotated_csv
"Supplementary Data 1.csv" --tiab_source "Supplementary Data 1.csv" --out_csv anno.csv`); its
labels are already corrected. `screen_preds.csv` (`pmid`, `pred`) comes from running the screen
over the test abstracts.

```bash
python -m litdd.training.attach_tiab --tiab_source anno.csv --out_csv data/annotated_pmid_tiab.csv
python -m litdd.training.final_traintest_dataset --annotated_csv data/annotated_pmid_tiab.csv --out_dir data --group_col pmid
python -m litdd.evaluation.build_llm_eval_shards --dataset_dir data/ds_test \
  --anno_csv anno.csv --g2p_csv G2P_DD.csv --screen_preds screen_preds.csv --out_dir fixture/
python -m litdd.evaluation.complete_test_pairs --anno_csv anno.csv --fixture_dir fixture/ --g2p_csv G2P_DD.csv
# steps 2-4 above on fixture/shards/annotated_test.parquet, writing fixture/candidates.parquet,
# fixture/llm_shards/ and fixture/out/
python -m litdd.evaluation.llm_adjudication_eval --llm_parquet "fixture/out/*__llm.parquet" \
  --gold_csv fixture/gold.csv --pairs_csv fixture/pairs_full.csv --g2p_csv G2P_DD.csv \
  --out_prefix fixture/out/eval --label myrun
python -m litdd.evaluation.stage_confusion_matrices --gold_csv fixture/gold.csv \
  --llm_parquet "fixture/out/*__llm.parquet" --g2p_csv G2P_DD.csv --out_csv stages.csv
```

The released pipeline scores precision 0.865, recall 0.806 and F1 0.835 on the test split
(2,731 abstracts, 645 with a curated entry; exact-set match end to end); the per-stage
confusion matrices are Table 1 of the paper.

### Datamap figure

The figure embeds each mapped paper with the MedEmbed-large-v0.1 bi-encoder (the default of
`embed_papers`); the embedding is used only for the figure and plays no part in the mapping.
Pass `--deterministic` to `layout_clusters` for a seeded layout.

```bash
python -m litdd.viz.embed_papers --map_csv results/litdd_pubmed2026_final_v6.csv \
  --text_parquet llm_all.parquet --out_parquet embeddings.parquet          # GPU
python -m litdd.viz.layout_clusters --in_parquet embeddings.parquet \
  --out_parquet clusters_and_viz.parquet                                    # GPU with cuML, else CPU
python -m litdd.viz.datamap_plot --clusters_parquet clusters_and_viz.parquet \
  --g2p_file G2P_DD.csv --mondo_owl mondo.owl --out_static datamap.png --out_interactive datamap.html
```

## Training the screen

```bash
./run_pipeline.sh --demo    # see Demo above
python -m litdd.training.attach_tiab --tiab_source <"Supplementary Data 1.csv" or dl/parquet_download_files> \
  --out_csv data/annotated_pmid_tiab.csv
./run_pipeline.sh --full    # full annotated set, released base model, grid of Supplementary Table 2, GPU
```

Both modes run the group-stratified split (`litdd.training.final_traintest_dataset`), a
cross-validated hyperparameter search on the training portion (`cv_hp_search_bert`) and one
refit with a single evaluation on the test portion (`bert_finetune`). The released checkpoint
was trained by `litdd.training.finetune_seeds` on the training set described in
`litdd/training/README.md` (the annotated split, the confirmed worksheet papers, the curated
positives on training-fold genes and 20,000 corpus negatives). `litdd.evaluation.run_bert_benchmark`
fine-tunes the baseline encoders with the same protocol.

## Citation

See `CITATION.cff`. Archived releases: https://doi.org/10.5281/zenodo.17779641.

## Licence

Code: MIT (see `LICENSE`). Data in `data/supplementary_data/` and `data/source_data/`: CC BY 4.0,
except Supplementary Data 3 (the G2P panel, under the EMBL-EBI terms of use).
