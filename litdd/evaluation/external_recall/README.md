# External-recall evaluation

Measures the released map's PMID-retrieval recall per disease (G2P id) against external
curated literature, and categorises the misses. Sources: the pre-mined DDG2P publications,
HPOA, and ClinGen case-level evidence.

Ground truth is restricted to disorders in the DDG2P export the pipeline was built on and to
leaf MONDO terms (single gene-diseases rather than grouping terms), excludes training and
test PMIDs (`--exclude_pmids`), and excludes papers the pipeline cannot retrieve (the screen
keeps publication years after 1980, so `--min_year 1981`).

## Scripts

| Script | Purpose |
|---|---|
| `build_truthsets.py` | Assemble `(g2p_id, pmid)` truth from the pre-mined DDG2P publications, HPOA (via OMIM) and ClinGen case-level exports (by MONDO), restricted to leaf MONDOs (`--mondo_json`). |
| `fetch_pmid_meta.py` | NCBI esummary: year, publication types and title for truth PMIDs absent from the pipeline output. |
| `measure_recall.py` | Per-disease micro and macro recall, over all truth PMIDs and over screen-positive ones, with miss categories. |
| `characterise_misses.py` | Miss categories joined to NCBI publication types (in scope versus review, editorial and similar). |
| `bert_negative_gene_check.py` | Whether the causative gene is mentioned in the screen-negative misses. |
| `build_external_positives.py` | Curated positives with their abstracts and a gene-fold split, as a training pool for the screen. |

## Run

```bash
python -m litdd.evaluation.external_recall.build_truthsets \
  --ddg2p G2P_DD.csv --mondo_backfill G2P_DD_newer.csv --hpoa phenotype.hpoa \
  --clingen_exports clingen_csv_exports/ --mondo_json mondo.json \
  --exclude_pmids annotated_tiab.csv --out_dir external_recall/

python -m litdd.evaluation.external_recall.fetch_pmid_meta \
  --pmids external_recall/bert_negative_pmids.txt --out external_recall/bert_negative_meta.csv

python -m litdd.evaluation.external_recall.measure_recall \
  --truthsets external_recall/truthsets.csv --litdd_map results/litdd_pubmed2026_final_v6.csv \
  --complete_df llm_all.parquet --pmid_years external_recall/bert_negative_meta.csv \
  --min_year 1981 --out_dir external_recall/
```

`mondo.json` is the MONDO obographs export (`purl.obolibrary.org/obo/mondo.json`).

Outputs: `recall_summary.csv` (source, scope, micro and macro recall) and
`miss_categories.csv` (`litdd_bert_negative`, `llm_no_match`, `mapped_other`,
`not_in_final_map`).
