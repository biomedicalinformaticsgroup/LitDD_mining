# External-recall evaluation (Reviewer 3 R3.4 / Reviewer 2 C1/C2)

Measures LitDD PMID-retrieval recall, **per disease (G2P ID)**, against external curated
literature, and categorises the misses. Reported sources: **pre-mined DDG2P publications**,
**HPOA**, and **ClinGen case-level evidence**.

Ground truth is restricted to disorders in the **August-2025 DDG2P** (the version LitDD is
built on), to **leaf MONDOs** (single gene-diseases, not broad grouping terms — see
`NOTES.md`), excludes **train/test** PMIDs (`--exclude_pmids`), and excludes papers LitDD
could not retrieve (BERT filters `pubdate > 1980`, so `--min_year 1981`).

## Scripts
| Script | Purpose |
|---|---|
| `build_truthsets.py` | Assemble `(g2p_id, pmid)` truth from premined (DDG2P `publications`), HPOA (via OMIM; multi-PMID refs parsed), ClinGen case-level (`genetic_evidence_*` exports, by MONDO). Restricts to leaf MONDOs (`--mondo_json`); MONDO backfilled by g2p-id from a newer DDG2P export (MONDO only). |
| `fetch_pmid_meta.py` | NCBI esummary (POST) → year + publication types + title for BERT-negative truth PMIDs (for `--min_year` and the scope/pubtype characterisation). |
| `measure_recall.py` | Per-disease micro & macro recall under `deployed` vs `relaxed` (gene filter off) variants, scope = `all` vs `bert_positive`; miss categories (`litdd_bert_negative`/`llm_no_match`/`mapped_other`/`below_score`/`gene_filtered`). |
| `characterise_misses.py` | Miss anatomy by category + NCBI publication type (in-scope vs review/editorial), to show the recall gap is a scope/BERT boundary rather than a ranking failure. |

## Run
```bash
REF=/path/to/clean_pipeline ; CD=/path/to/comparison_data
uv run python benchmarking/external_recall/build_truthsets.py \
  --ddg2p "$REF/annotate_pubmed/data/G2P_DD_2025-08-04.csv" \
  --mondo_backfill revision/G2P_DD_2026-06-24.csv \
  --hpoa "$CD/phenotype.hpoa" \
  --clingen_exports revision/clingen/clingen_csv_exports \
  --mondo_json revision/mondo.json \
  --exclude_pmids "$REF/train_test/annotated_tiab.csv"

uv run python benchmarking/external_recall/fetch_pmid_meta.py \
  --pmids revision/external_recall/bert_negative_pmids.txt \
  --out revision/external_recall/bert_negative_meta.csv     # BERT-negative truth PMIDs

uv run python benchmarking/external_recall/measure_recall.py \
  --litdd_map annotate_pubmed/ddg2p_pubmed_map.csv \
  --complete_df "$REF/annotate_pubmed/data/pipeline_df_complete.parquet" \
  --pmid_years revision/external_recall/bert_negative_meta.csv --min_year 1981
```
`mondo.json` is the MONDO obographs export (`purl.obolibrary.org/obo/mondo.json`); not
committed (large) — download into `revision/`.

## Headline (deployed, all-papers denominator, min_year 1981)
Reproduces manuscript Table 6: premined 0.68/0.72, HPOA 0.69/0.72, ClinGen 0.71/0.72
(micro/macro); combined 0.66/0.70. Relaxing the gene-mention filter (`relaxed`) adds only
~1–4 points, so the filter's recall cost is small. The dominant miss is
`litdd_bert_negative` — papers LitDD's BERT (run over all PubMed) classified negative; ~29%
of these do not mention the causative gene even in the abstract, consistent with the
deliberate exclusion of papers lacking molecular confirmation (R3.4). Outputs go to the
gitignored `revision/external_recall/`.
