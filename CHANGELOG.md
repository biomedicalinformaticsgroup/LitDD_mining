# Changelog

Dated changes to the pipeline, with the measurement that motivated each. Measurements refer
to the 2026 PubMed baseline and the 2026-06-24 G2P developmental-disorder export unless stated.

## 2026-09 repository reduction to the released pipeline

- The public repository now implements only the released (v6) pipeline: vLLM screen, HGNC
  identifier gene gate, contextualised-thread adjudication with GPT-OSS-20B, candidate
  membership clean. The cross-encoder stage, the symbol-string gene gate, the disease-alias
  stop list of the gate, the original decision rubric and every adjudication ablation switch
  were removed. Copies are kept in the author's working copy. Results of the removed
  configurations remain in `supplementary/llm_ablation_ledger.csv`.
- The datamap figure is built by `litdd.viz.embed_papers` (MedEmbed-large-v0.1 bi-encoder,
  first-token pooling, used for this figure only), `litdd.viz.layout_clusters` (UMAP and HDBSCAN)
  and `litdd.viz.datamap_plot` (Mondo labels). A layout from the screen model's own encoder was
  tried and left 78 percent of papers unclustered against 35 percent for the bi-encoder.
- `supplementary/original_vs_new_pipeline.csv` was removed; the six ledger rows scored on the
  submitted pipeline's candidates were dropped from `supplementary/llm_ablation_ledger.csv`.
- Adjudication inputs carry `candidates` (list of G2P ids) instead of `top5_cross` (list of
  label and score); the rendered candidate text column is `candidate_text`. A new stage,
  `litdd.pipeline.build_llm_shards`, writes the adjudication shards from the gate output.
- The contextualised-thread builder, previously outside the repository, is
  `litdd.pipeline.build_context_threads`. Its output for the 2026-09-10 all-panel export is
  `data/context_threads_G2P_all_2026-09-10.json`. Regenerated blocks match the released file
  except in the order of the `Disease Synonyms` line, which the previous builder emitted in
  hash order.
- `final_data_clean` reads every GeneID of a multi-id PubTator cell when listing the genes
  mentioned in a NO MATCH abstract; previously only the first.
- Scripts are run as modules (`python -m litdd.<package>.<module>`); the `sys.path` edits are
  gone and pytest uses `pythonpath = ["."]`.
- `results/` holds only the released map (`litdd_pubmed2026_final_v6.csv`); the maps published
  with the submitted pipeline remain in the repository history.

## 2026-09-17 released configuration settled

- Released map: 86,513 mappings over 72,939 papers and 2,826 entries. Test split (2,731
  abstracts, 645 curated, exact-set match end to end): precision 0.865, recall 0.806, F1 0.835
  (TP 520, FP 81, FN 125, TN 2,048). Development split 0.813 / 0.825 / 0.819. Held-out curated
  sets: end-to-end recall 0.903 raw, 0.911 in scope.
- The gate was re-run with HGNC-identifier resolution; adjudications from the previous run
  were reused only for abstracts whose candidate set was identical, and 3,827 changed or new
  abstracts were re-adjudicated.

## 2026-09-11 gene gate: symbol fallback blocklist narrowed

- The verbatim-symbol fallback (used only where PubTator3 annotated no panel gene) blocked 13
  panel symbols outright. Measured on the 179,748 screen-positive abstracts (25,842 without any
  PubTator annotation): matching is uppercase-only and word-bounded, so the English-word
  collisions the list was built for never matched; for 9 of the 13 genes every
  fallback-eligible match was the gene (ATM 8/8, GAN 5/5, SON 6/6, SHOX 1/1, REST 1/1; KIT,
  NODAL, PIGN, PIGS no matches). CAD-related epileptic encephalopathy had lost its entire mined
  set to the block.
- CAD, SET and STAR collide with acronyms only when a competing expansion is present; they are
  now resolved by context rules (document-level expansions; occurrence-level "SET domain",
  "SET binding factor", "sequencing KIT"). The rules were fitted to the 81 abstracts the
  narrowing newly admits; "SET domain" and related protein terms accounted for 11 of the 17
  false matches the document rule alone let through.
- TAT stays blocked: 4 of its 5 fallback-eligible matches were DNA codons.

## 2026-09 gene gate: HGNC identifier resolution replaces symbol strings

- The previous gate resolved a PubTator3 GeneID to its NCBI symbol and matched the string
  against G2P gene symbols and previous symbols. This let one gene's current symbol hit another
  gene's former symbol (OPA1 to MED12, AR to FDXR, FLG to FGFR1) and missed genes whose NCBI
  symbol differs from G2P's (TRNL1 versus MT-TL1). 4,474 candidates in the corpus run were
  wrong-gene by this route, 203 in the released v5 map.
- Every route now resolves to an HGNC id and joins G2P on its `hgnc id` column. G2P previous
  gene symbols are no longer used: 1,185 of 7,210 are not HGNC previous or alias symbols of
  the gene, and 103 are another gene's approved symbol.
- Disease names and abbreviations are excluded as gene evidence from the MONDO lexicon
  (previously a stop-list TSV built from HGNC and MONDO).

## 2026-09-05 corpus definition: correction RefTypes no longer excluded

- `ErratumIn` and `ErratumFor` were removed from the RefType exclusion set of
  `dedupe_pmids`. Excluding `ErratumIn` dropped every paper that has a published correction:
  340,217 eligible records (1.08 percent of the corpus), including gene-discovery reports, and
  294 of the 7,644 curated external-truth papers were never screened, the second-largest cause
  of recall misses (22.7 percent). Correction notices themselves remain excluded by publication
  type (D016425).
- Publication-type and RefType exclusions agree on retractions (RetractionIn 34,178 versus
  D016441 34,180, symmetric difference 2). The RefType pass adds `ExpressionOfConcernIn`
  (3,526 not caught by publication type) and the superseded versions of republished articles
  (`RepublishedIn` 1,400, `RetractedandRepublishedIn` 48).

## 2026-09-02 adjudication and evaluation

- The candidate-membership check in `final_data_clean` replaced the gene-mention re-test. On
  the 2026 corpus run the re-test dropped 9,006 mappings, of which 8,989 (99.8 percent) were
  candidates the gate had admitted.
- Gene gate mentions are verified in the title and abstract: the bulk PubTator file mixes
  text annotations with database cross-references and full-text annotations, inflating 3
  percent of development-split abstracts to 20 to 104 genes.
- Scoring rule: two entries of one gene with the same disease name differing only in
  monoallelic X heterozygous versus hemizygous are interchangeable for evaluation.

## 2026-09-01 annotation corrections

- `data/annotation_corrections.csv` records the reviewed relabellings, additions and
  removals applied to the annotation before any fixture is built (restored sibling negatives,
  the PURA relabelling, co-reported pairs, removal of a cancer-predisposition entry).

## 2026-08 screen

- Released checkpoint: seed 44, trained on the annotated set plus confirmed positives,
  external curated positives and 20,000 corpus negatives. Hugging Face
  `tmy100000001/LitDD_BERT`, branch `main`; the earlier seed-42 checkpoint is tag `v1-seed42`.
- Corpus negatives were added because the high-recall screen trained at 51.8 percent positive
  fired on 19.46 percent of random PubMed records against a deployment prevalence of 1 to 2
  percent; the in-domain test set (25 percent positive) could not detect this. Fire rates:
  0.51 percent after retraining, 3.4 to 3.6 percent in-domain false-positive rate for every
  checkpoint.
- Screen training and evaluation standardised on fp32: the checkpoint had been trained in bf16
  while the baseline benchmark ran in fp32, so the shipped model could not be compared with its
  own base. Under vLLM, bf16 differs from fp32 on 1 of 2,779 test rows (F1 0.9213 versus
  0.9206, positive rate 26.27 versus 26.23 percent) and is about 1.3 times faster; fp32 remains
  the default.
- Screen inputs use the full ModernBERT context (8,192 tokens); the earlier 512-token cap,
  inherited from the BERT-large base, truncated about 1 percent of abstracts.
- The baseline benchmark fine-tunes every baseline with the same protocol as the released
  screen; an earlier draft scored baselines with an untrained classification head.

## 2026-08 adjudication stage engineering

- Work is split across workers by row rather than by shard file: with 4 files and 8 workers,
  workers 4 to 7 received no rows.
- Generation is one vLLM call per checkpoint window instead of fixed 12-prompt batches, which
  had defeated continuous batching (about 186 output tokens per second for a 14B model on an
  A100). Prefix caching is enabled because the rubric is a fixed prefix of about 5,000
  characters.
- Resume: a shard interrupted on a preemptible node continues from its checkpoint instead of
  restarting at row 0.
- Adjudication moved from DeepSeek-R1-Distill-Qwen-14B (raw completion prompt) to GPT-OSS-20B
  through the chat template with a revised decision rubric; candidates are every entry of the
  detected genes across all G2P panels, contextualised from the all-panel export, with answers
  restricted to the developmental-disorder panel.

## 2026-08 candidate label rendering

- Training and inference had rendered the G2P entry label differently (a column looked up
  under a name absent from every export was always blank at inference; missing values rendered
  as `nan` versus empty; MIM numbers as floats versus integers). `litdd.threads` renders the
  label once, resolving columns by name with aliases for the 2025 and 2026 export layouts.

## 2026-07 corpus ingestion

- `pubmed_parser` emits no row for `<DeleteCitation>` entries (its `delete` flag was true for
  0 of 45,056,462 records), so withdrawn PMIDs are extracted from the raw XML by
  `extract_deleted_pmids`; 7,408 withdrawn PMIDs in the 2026 update files, 5,025 still present
  after conversion.
- About 9 percent of PubMed records appear in more than one baseline or update file;
  `dedupe_pmids` keeps the latest occurrence. The screen runs over 31.57M records after
  de-duplication and retraction removal instead of 40.4M.
- Records with a delimited language field (`eng;spa` 45,189, `eng;por` 31,482, and others:
  134,817 records) are excluded; only `eng` records published after 1980 are screened.
