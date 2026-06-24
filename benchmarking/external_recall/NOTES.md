# External-recall: internal notes (NOT for the manuscript)

## ClinGen is excluded from the reported recall

`measure_recall.py` computes ClinGen recall for our own analysis but it is **not reported**
in the paper (`REPORTABLE = ("premined", "hpoa")`; the `combined` row excludes it). Two
reasons:

1. **Web-scraped, not a citable release.** The ClinGen evidence PMIDs come from a scrape
   (`clingen_pmid_df.p`), not a versioned, dated, redistributable file like HPOA's
   `phenotype.hpoa` (release 2025-03-03) or DDG2P's own `publications`. It is not a stable
   benchmark we can cite or that a reader could reproduce.

2. **Out of LitDD's scope.** ClinGen gene–disease *validity* evidence is dominated by
   **functional / mechanistic** literature — mouse/zebrafish models, molecular-mechanism
   studies, population genomics — not the human **case reports / case series** LitDD is
   built to retrieve. The misses are recent (median year 2010; only ~18% pre-2000), so this
   is a paper-*type* mismatch, not an age effect. Examples of missed ClinGen evidence:
   - mouse models (e.g. *FgfR3P244R cranial base*; *Mouse Models of Methylmalonic Aciduria*)
   - zebrafish functional studies (*myotubularin T-tubule disorganization*)
   - mechanism papers (*RSK2 in mammalian neurogenesis*)
   - population sequencing (*whole-genome sequencing of the Icelandic population*)
   LitDD's BERT screen correctly rejects these, so a low ClinGen "recall" (~0.34) reflects
   a deliberate scope boundary, not a retrieval failure.

For the manuscript we therefore report recall against **pre-mined DDG2P publications** and
**HPOA** (both clinical, case-report-weighted literature that matches LitDD's target), and
characterise ClinGen separately only to evidence the scope argument (R3.4).
