# Running the LitDD pipeline

The pipeline maps PubMed titles and abstracts to Gene2Phenotype (G2P) developmental-disorder
entries in four stages, with two preparatory steps. Every stage is a module under `litdd/`
and is run as `python -m litdd.<package>.<module>`; each prints its options with `--help`.

    PubMed TIAB parquet shards
      -> 1. screen        litdd.pipeline.bert_predict_vllm     HF tmy100000001/LitDD_BERT
      -> 2. gene gate     litdd.pipeline.gene_candidates       HGNC-identifier resolution
      -> 3. shards        litdd.pipeline.build_llm_shards      adjudication inputs
      -> 4. adjudication  litdd.pipeline.llm_map               openai/gpt-oss-20b, contextualised threads
      -> 5. clean         litdd.pipeline.final_data_clean      candidate-membership check

## Inputs

| Input | Source |
|---|---|
| G2P developmental-disorder panel CSV (`G2P_DD.csv`) | https://www.ebi.ac.uk/gene2phenotype/downloads |
| G2P all-panel CSV (`G2P_all.csv`) | same page, every panel; used for the context threads and the adjudication stage |
| `gene2pubtator3.gz` | https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/ |
| `hgnc_complete_set.txt` | https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt |
| `gene_info.gz` (Homo sapiens) | https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz |
| `mondo.obo`, `hp.obo` | http://purl.obolibrary.org/obo/mondo.obo, http://purl.obolibrary.org/obo/hp.obo |
| PubMed baseline and update files | `python -m litdd.pipeline.download_pubmed`, then `pubmed_to_parquet` and `dedupe_pmids` |

The models (`tmy100000001/LitDD_BERT`, branch `main`, release `v2.5`; `openai/gpt-oss-20b`,
revision `6cee5e81` in the released run) are downloaded from Hugging Face by vLLM. On compute nodes without internet access, stage them into a Hugging Face cache
first and point `HF_HOME` at it.

## Preparatory step: context threads

The adjudication stage shows each candidate G2P entry as a multi-line block built from the
all-panel export, MONDO and HPO. The released block set is
`data/context_threads_G2P_all_2026-09-10.json`. To rebuild it for a new export:

```bash
python -m litdd.pipeline.build_context_threads --g2p_csv G2P_all.csv \
  --mondo_obo mondo.obo --hp_obo hp.obo --out_json context_threads.json
```

The builder removes disease-name aliases from each entry's former gene symbols using
`litdd/pipeline/data/disease_name_aliases.tsv`, which `litdd.pipeline.build_disease_name_aliases`
regenerates from HGNC and MONDO.

## Run on a corpus (GPU)

Container: `ghcr.io/biomedicalinformaticsgroup/litdd_mining` (built by `.github/workflows/image.yml`
from `containers/Dockerfile`; vLLM 0.23.0). Use the SHA-tagged image in job manifests.

```bash
# 1. screen. Pass the Hugging Face id; the positive rate on random PubMed is about 0.5%.
python -m litdd.pipeline.bert_predict_vllm --model tmy100000001/LitDD_BERT \
  --input_dir <parquet shards with pmid,tiab,languages,pubdate> --processed_dir screened/ \
  --keep_parquet <download_dir>/pmid_keep.parquet
python -m litdd.pipeline.build_bert_positives --processed_dir screened/ --out_path screened/bert_positive.parquet

# 2. gene gate. --audit_prefix writes every HGNC symbol and name considered and why it was
#    kept or excluded, with per-rule counts.
python -m litdd.pipeline.gene_candidates \
  --input_parquet screened/bert_positive.parquet --g2p_csv G2P_DD.csv \
  --gene2pubtator gene2pubtator3.gz --hgnc hgnc_complete_set.txt --mondo_obo mondo.obo \
  --audit_prefix gate_audit/corpus --out_parquet candidates.parquet

# 3. shards for the adjudication stage
python -m litdd.pipeline.build_llm_shards --candidates_parquet candidates.parquet \
  --out_dir shards/ --num_shards 8

# 4. adjudication (one worker per shard index, or one worker for all shards)
python -m litdd.pipeline.llm_map --shards_dir shards/ --out_dir out/ \
  --llm_model openai/gpt-oss-20b --temperature 0.0 --top_p 1.0 --seed 0 \
  --reasoning_effort medium --max_model_len 32768 --save_every 5000 \
  --context_json data/context_threads_G2P_all_2026-09-10.json \
  --context_drop_fields "Disease Definition,Phenotypes" \
  --panel_siblings_csv G2P_all.csv --final_panel_csv G2P_DD.csv

# 5. clean. --llm_file takes one parquet: concatenate the adjudication shards first.
python -m litdd.pipeline.final_data_clean --llm_file llm_all.parquet --g2p_file G2P_DD.csv \
  --candidates_parquet candidates.parquet --gene2pubtator gene2pubtator3.gz --gene_info gene_info.gz \
  --output_csv final.csv --no_match_csv nomatch.csv
```

`final.csv` has the columns `PMID` and `G2P_IDs`, one row per (abstract, entry) pair.
`nomatch.csv` lists the abstracts the adjudicator mapped to no entry, with the genes mentioned
in each; some are candidate gene-disease relationships not yet in G2P. The released map is
`results/litdd_pubmed2026_final_v6.csv`.

## Evaluate a run on the annotated set

The annotated split is turned into a fixture, run through the gate and the adjudicator, and
scored:

```bash
python -m litdd.evaluation.apply_annotation_corrections --corrections data/annotation_corrections.csv \
  --anno_csv g2p_id_tiab_anno_df_FINAL.csv --anno_out anno_corrected.csv \
  --pmid_csv data/annotated_pmid.csv --g2p_csv G2P_DD.csv
python -m litdd.training.final_traintest_dataset --annotated_csv data/annotated_pmid.csv --out_dir data --group_col pmid
python -m litdd.evaluation.build_llm_eval_shards --dataset_dir data/ds_test \
  --anno_csv anno_corrected.csv --g2p_csv G2P_DD.csv --screen_preds screen_preds.csv --out_dir fixture/
python -m litdd.evaluation.complete_test_pairs --anno_csv anno_corrected.csv --fixture_dir fixture/ --g2p_csv G2P_DD.csv
python -m litdd.pipeline.gene_candidates --input_parquet fixture/shards/annotated_test.parquet ... --out_parquet fixture/candidates.parquet
python -m litdd.pipeline.build_llm_shards --candidates_parquet fixture/candidates.parquet --out_dir fixture/llm_shards/
python -m litdd.pipeline.llm_map --shards_dir fixture/llm_shards/ --out_dir fixture/out/ ...   # as in step 4
python -m litdd.evaluation.llm_adjudication_eval --llm_parquet "fixture/out/*__llm.parquet" \
  --gold_csv fixture/gold.csv --pairs_csv fixture/pairs_full.csv --g2p_csv G2P_DD.csv \
  --out_prefix fixture/out/eval --label myrun
python -m litdd.evaluation.stage_confusion_matrices --gold_csv fixture/gold.csv \
  --llm_parquet "fixture/out/*__llm.parquet" --g2p_csv G2P_DD.csv --out_csv stages.csv
```

`screen_preds.csv` (columns `pmid`, `pred`) comes from running the screen over the fixture
abstracts. The per-stage confusion matrices of the released pipeline on the test split are in
`supplementary/stage_confusion_matrices.csv`; the configurations evaluated during the
revision are listed in `supplementary/llm_ablation_ledger.csv`.

## Datamap figure

The figure embeds each mapped paper with the MedEmbed-large-v0.1 bi-encoder (the default of
`embed_papers`). The embedding is used only for the figure and plays no part in the mapping.

```bash
python -m litdd.viz.embed_papers --map_csv results/litdd_pubmed2026_final_v6.csv \
  --text_parquet llm_all.parquet --out_parquet embeddings.parquet          # GPU
python -m litdd.viz.layout_clusters --in_parquet embeddings.parquet \
  --out_parquet clusters_and_viz.parquet                                    # GPU with cuML, else CPU
python -m litdd.viz.datamap_plot --clusters_parquet clusters_and_viz.parquet \
  --g2p_file G2P_DD.csv --mondo_owl mondo.owl --out_static datamap.png --out_interactive datamap.html
```

## Notes

* Labels: apply `data/annotation_corrections.csv` before building a fixture; the raw
  annotation file lacks the sibling negatives of positive abstracts and contains entries
  that were later relabelled or removed.
* Screen checkpoint: pass the Hugging Face id to `--model`. Loading the checkpoint from a
  local directory under vLLM 0.23 returned a constant probability for every abstract.
* G2P versions: use one export throughout an evaluation. The context-thread JSON must be
  built from the same all-panel export that `--panel_siblings_csv` names.
