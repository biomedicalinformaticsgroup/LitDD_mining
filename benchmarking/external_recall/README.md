# External-recall evaluation (Reviewer 3 R3.4 / Reviewer 2 C1/C2)

Measures LitDD PMID-retrieval recall, **per disease (G2P ID)**, against external curated
literature, and categorises the misses. Reported sources: **pre-mined DDG2P publications**
and **HPOA** (clinical, case-report-weighted). ClinGen is computed but not reported — see
`NOTES.md`.

All ground truth is restricted to disorders in the **August-2025 DDG2P** (the version LitDD
is built on) and to papers LitDD could retrieve (the BERT step filters `pubdate > 1980`, so
`--min_year 1981` drops older papers from the denominator).

## Scripts
| Script | Purpose |
|---|---|
| `build_truthsets.py` | Assemble `(g2p_id, pmid)` truth from premined (DDG2P `publications`), HPOA (via OMIM; multi-PMID references parsed), ClinGen (via gene+MONDO). MONDO backfilled by g2p-id from a newer DDG2P export (MONDO only). |
| `fetch_pmid_meta.py` | NCBI esummary (POST) → year + publication types + title for truth PMIDs not in the corpus (for `--min_year` and the scope/pubtype characterisation). |
| `measure_recall.py` | Per-disease micro & macro recall under `deployed` vs `relaxed` (gene filter off) variants, scope = `all` vs `in_corpus`; miss categories (`not_in_corpus`/`llm_no_match`/`mapped_other`/`below_score`/`gene_filtered`). |

## Run
```bash
REF=/path/to/clean_pipeline ; CD=/path/to/comparison_data
uv run python benchmarking/external_recall/build_truthsets.py \
  --ddg2p "$REF/annotate_pubmed/data/G2P_DD_2025-08-04.csv" \
  --mondo_backfill revision/G2P_DD_2026-06-24.csv \
  --hpoa "$CD/phenotype.hpoa" --clingen_pickle "$CD/clingen_pmid_df.p" \
  --clingen_summary "$CD/Clingen-Gene-Disease-Summary-2025-03-17.csv"

uv run python benchmarking/external_recall/fetch_pmid_meta.py \
  --pmids revision/external_recall/not_in_corpus_pmids.txt \
  --out revision/external_recall/not_in_corpus_meta.csv     # needs the not-in-corpus PMID list

uv run python benchmarking/external_recall/measure_recall.py \
  --litdd_map annotate_pubmed/ddg2p_pubmed_map.csv \
  --complete_df "$REF/annotate_pubmed/data/pipeline_df_complete.parquet" \
  --pmid_years revision/external_recall/not_in_corpus_meta.csv --min_year 1981
```

## Headline (deployed corpus, all-papers denominator)
premined ≈ 0.65, HPOA ≈ 0.62 (→ 0.72 with the gene filter relaxed); combined ≈ 0.63 → 0.69.
Restricting to papers in the corpus (`in_corpus`): combined 0.74 → 0.81 relaxed. The
dominant miss is `not_in_corpus` — curated PMIDs the BERT screen never saw (largely
out-of-scope non-case-report citations), which is the R3.4 characterisation.
Outputs are written to the gitignored `revision/external_recall/` (contain PMID lists).
