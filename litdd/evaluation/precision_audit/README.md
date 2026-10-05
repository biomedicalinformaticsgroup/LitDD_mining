# Precision audit and inter-annotator agreement

Scripts for measuring the precision of the released map by a blinded, stratified manual
audit, and for quantifying inter-annotator agreement. They use only numpy and pandas.
Worksheets and keys contain abstracts and are annotator-facing; write them outside the
repository.

## Scripts

| Script | Purpose |
|---|---|
| `sample_audit.py` | Draw a blinded, stratified sample of (PMID, G2P entry) mappings from the adjudication output (strata: recency, disease volume, gene multiplicity). Writes `audit_worksheet.csv` (annotator A), `audit_worksheet_overlap.csv` (subset for annotator B) and `audit_key.csv` (strata, hidden from annotators). |
| `sample_trainlabel_iaa.py` | Draw a sample of the training annotations for a second annotator to re-label. Writes `trainlabel_iaa_worksheet.csv` and `trainlabel_iaa_key.csv`. |
| `cascade_funnel.py` | Count abstracts, candidate pairs and mappings surviving each pipeline stage. |
| `score_audit.py` | After annotation: precision overall and per stratum with Wilson 95% intervals, the implied false-positive count at corpus size, error categories, and Cohen's kappa for both agreement exercises. |

## Run order

```bash
python -m litdd.evaluation.precision_audit.sample_audit \
    --input llm_all.parquet --g2p_file G2P_DD.csv --out_dir audit/ \
    --cutoff_year 2025 --min_post_cutoff 80

python -m litdd.evaluation.precision_audit.sample_trainlabel_iaa \
    --annotated_csv data/annotated_pmid.csv --out_dir audit/

python -m litdd.evaluation.precision_audit.cascade_funnel \
    --complete_df llm_all.parquet --candidates_parquet candidates.parquet \
    --final_map results/litdd_pubmed2026_final_v6.csv --out_dir audit/

# annotators fill in the worksheets

python -m litdd.evaluation.precision_audit.score_audit --audit_dir audit/ \
    --corpus_n <number of mappings in the released map> --cutoff_year 2025
```

### Knowledge cutoff

`--cutoff_year` splits the audited mappings by publication year so precision on records
published after the adjudication model's training data can be compared with precision on
earlier records. The sampler guarantees a minimum number of post-cutoff records so that
comparison has a usable interval.

## Annotator instructions

**Audit worksheet (`audit_worksheet.csv`, annotator A; the overlap subset also by annotator B).**
Each row is one mapping the pipeline emitted: an abstract (`title`, `abstract`) and the G2P
entry it was assigned (`assigned_disease`, `assigned_gene`, `assigned_candidate_text`). Fill:

- `verdict`: `correct` (the abstract concerns this G2P gene-disease entry), `incorrect`, or
  `uncertain`.
- `error_category` (only if `incorrect`): one of `wrong_gene`, `wrong_allelic_requirement`,
  `wrong_mechanism`, `somatic_only`, `non_human_only`, `acronym_gene_confusion`,
  `cnv_snv_confusion`, `no_molecular_confirmation`, `wrong_disease_same_gene`, `other`.
- `notes`: free text.

Annotators do not see the strata, which are kept in `audit_key.csv`.

**Training-label agreement (`trainlabel_iaa_worksheet.csv`, the second annotator).**
Each row is an abstract and a candidate G2P entry. Fill `relevant`: `1` if the abstract
supports mapping to this entry, `0` if not, `uncertain` if unclear, without sight of the
original label (kept in the key).

## Notes

- Precision excludes `uncertain` verdicts, which are reported separately.
- For a population-weighted overall precision, reweight the per-stratum precisions by each
  stratum's share of the full map; the sampler oversamples small cells.
