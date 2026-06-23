# Deployed-corpus precision audit + inter-annotator agreement

Tooling for Reviewer 2 B1 (precision at deployment scale, R2-P1/P2/P3 + R2-A1) and
Reviewer 3 R3.10. The headline 0.83 precision is from a ~26%-positive balanced test set;
these scripts measure precision on the **deployed corpus** via a blinded, stratified
manual audit, and quantify inter-annotator agreement.

All worksheet/key outputs default to `revision/precision_audit/` (gitignored) because they
contain real abstracts and are annotator-facing. Scripts use only numpy/pandas (CPU).

## Scripts

| Script | Purpose |
|---|---|
| `sample_audit.py` | Draw a blinded, stratified ~500 sample of `(PMID → G2P disease)` mappings (strata: cross-encoder confidence, recency, disease volume, gene multiplicity). Emits `audit_worksheet.csv` (annotator A), `audit_worksheet_overlap.csv` (~100 for annotator B → κ), and `audit_key.csv` (hidden scores/strata). |
| `sample_trainlabel_iaa.py` | Draw 100 from the ~13k training annotations for a second annotator to re-label (R3.10). Emits `trainlabel_iaa_worksheet.csv` + `trainlabel_iaa_key.csv`. |
| `cascade_funnel.py` | Count records surviving each pipeline stage (R2-P3) and optionally sample the mappings dropped by the gene/score gate (R2-C1/R3.4). |
| `score_audit.py` | After annotation: per-stratum + overall precision with Wilson 95% CIs, implied false-positive count at corpus scale, error-category breakdown (R2-D1/D2), and Cohen's κ for both IAA exercises. |

## Run order

```bash
# Reference data location (parquet/CSV not shipped — publisher/size); e.g.:
REF=/path/to/clean_pipeline

# 1. Deployed-corpus audit sample (blinded worksheet for annotator A + overlap for B)
uv run python benchmarking/precision_audit/sample_audit.py \
    --input "$REF/annotate_pubmed/data/final_tiab_mappings.parquet" \
    --g2p_file "$REF/annotate_pubmed/data/G2P_DD_2025-08-04.csv"

# 2. Training-label IAA sample (second annotator re-labels these)
uv run python benchmarking/precision_audit/sample_trainlabel_iaa.py \
    --annotated_csv "$REF/train_test/annotated_tiab.csv"

# 3. Cascade funnel (+ optional dropped-set worksheet)
uv run python benchmarking/precision_audit/cascade_funnel.py \
    --llm_map annotate_pubmed/pubmed_ddg2p_map.csv \
    --final_map annotate_pubmed/ddg2p_pubmed_map.csv \
    --bert_positive "$REF/annotate_pubmed/data/pubmed_bert_positive.parquet"

# --- annotators fill in the worksheets (see below) ---

# 4. Score everything
uv run python benchmarking/precision_audit/score_audit.py
```

## Annotator instructions

**Audit worksheet (`audit_worksheet.csv`, annotator A; overlap subset also by annotator B).**
Each row is one mapping the pipeline emitted: an abstract (`title`/`abstract`) and the G2P
disease it was assigned (`assigned_disease`, `assigned_gene`, `assigned_lgmde_thread`).
Fill:
- `verdict`: `correct` (the abstract genuinely concerns this G2P gene–disease entry),
  `incorrect`, or `uncertain`.
- `error_category` (only if `incorrect`): one of `wrong_gene`, `wrong_allelic_requirement`,
  `wrong_mechanism`, `somatic_only`, `non_human_only`, `acronym_gene_confusion`,
  `cnv_snv_confusion`, `no_molecular_confirmation`, `wrong_disease_same_gene`, `other`.
- `notes`: free text.
You are blinded to the model's confidence score and stratum (kept in `audit_key.csv`).

**Training-label IAA (`trainlabel_iaa_worksheet.csv`, the second annotator).**
Each row is an abstract + a candidate G2P disease. Fill `relevant`: `1` if the abstract
supports mapping to this disease, `0` if not, `uncertain` if unclear — independently of the
original label (which is hidden in the key). This reproduces the original binary annotation
task so κ vs the original labels can be computed.

## Notes
- Precision excludes `uncertain` (reported separately). `score_audit.py` also prints the
  implied false-positive count at the corpus size and a per-stratum table → Source Data.
- For a population-weighted overall precision, reweight the per-stratum precisions by each
  stratum's share of the full corpus (the audit oversamples small high-uncertainty cells).
