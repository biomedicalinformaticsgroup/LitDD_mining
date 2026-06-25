# External-recall: methodology notes (for the Methods text)

Decisions behind `build_truthsets.py` / `measure_recall.py`.

## Disease unit = G2P ID; recall universe = leaf MONDOs

A disease is a G2P entry (unique `g2p id`). External truth (premined / HPOA / ClinGen) is
matched to G2P at the disease level, and the universe is restricted to **leaf MONDOs** —
MONDO terms with no MONDO disease subclass.

- A grouping MONDO (e.g. `MONDO:0005021` "dilated cardiomyopathy", `MONDO:0044970` MT-TL1
  mitochondrial disorder, `MONDO:0018230` "skeletal dysplasia") aggregates curated
  literature spanning many genes; assigning it to the single G2P entry that carries it
  produces large spurious truth sets that the pipeline (correctly) maps to the specific
  sibling disease. The leaf rule drops these (338 grouping entries) and keeps single
  gene-diseases (e.g. SMC1A Cornelia de Lange, ARG1 argininemia, HRAS Costello).
- MONDO's explicit gene-grounding axiom — `has material basis in germline mutation in
  <GENE>` (RO:0004003) — is the ideal signal, **but it is absent for ~70% of leaf
  gene-diseases that should carry it** (a MONDO annotation gap), so using it literally would
  wrongly discard core single-gene diseases. The leaf-vs-grouping graph structure is used
  instead.
- A few newer G2P entries (`G2P031xx`, e.g. SEMA3A/LYSET "skeletal dysplasia") are specific
  gene-diseases but were annotated by DDG2P to a broad MONDO (no gene-specific MONDO exists
  yet); the leaf rule excludes them (we cannot match on a 119-child MONDO) — a DDG2P/MONDO
  annotation limitation.
- **53 leaf MONDOs still map to >1 G2P ID** (mono-/bi-allelic pairs of one disease); these
  are kept. Excluded entries are written to `excluded_grouping_mondos.csv`.

## ClinGen = case-level (genetic) evidence only

ClinGen recall uses the **genetic** evidence tables (`genetic_evidence_*tableExport.csv`:
variants / segregation / case_control), parsed from the `Reference(PMID)` column only (not
the free-text Explanation). The `experimental_evidence` tables (functional / model-organism /
mechanism) are excluded — out of LitDD's human-case-report scope, matching the manuscript's
"case level evidence" definition (Table 6). Using all ClinGen evidence collapses recall (the
functional papers are correctly rejected by the BERT screen).

## Train/test excluded; `publications` column only

`--exclude_pmids annotated_tiab.csv` removes train/test PMIDs so recall reflects only
held-out literature (standard generalisation evaluation; matches the manuscript's exclusion
of train/test manuscripts). The premined truth uses the DDG2P `publications` column **only**
— never `additional mined publications` (col 20 of the 2026 export), which is the LitDD
output fed back and is not independent.

## deployed ≈ relaxed ≈ Table 6

`deployed` (the shipped score≥0.9 + gene-in-TIAB map) and `relaxed` (gene filter off) give
near-identical recall — the gene-mention filter costs only ~1–4 points. Both reproduce the
manuscript's Table 6 (premined ~0.68/0.72, HPOA ~0.69/0.72, ClinGen ~0.71/0.72). An earlier
~8-point deployed shortfall was a bug: 8% of deployed-map rows carry several `;`-separated
G2P IDs in one cell and were not split (fixed in `mined_deployed`).
