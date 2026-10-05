# Design notes

Rationale for design decisions in the pipeline, with the measurements they rest on. The
code itself describes what it does; this file records why.

## Corpus definition

**Eligibility.** A record is screened when its language field is exactly `eng` and its
publication year is after 1980. Records whose language field is delimited (`eng;spa`,
`eng;por`, 134,817 records in the 2026 baseline) are not reliably English-language papers and
the screen was trained on monolingual English abstracts.

**Duplicates.** PubMed update files reissue records already present in the annual baseline
(about 9 percent of records). `dedupe_pmids` keeps the occurrence from the highest-numbered
file, which is the corrected version. A duplicated PMID would otherwise pass through every
stage twice.

**Withdrawn records.** `<DeleteCitation>` is a sibling of `<PubmedArticle>` holding bare
PMIDs, so an article-walking parser never emits a row for it and `pubmed_parser`'s `delete`
flag is never set. `extract_deleted_pmids` reads the raw XML instead.

**Retractions and corrections.** MEDLINE marks them in two places: MeSH publication types on
the notice (and, for retractions, on the retracted article), and `CommentsCorrections`
RefTypes that link the paper to its notice. Publication types and RefTypes agree on
retractions (RetractionIn 34,178 versus D016441 34,180). The RefType pass is kept for
expressions of concern about a paper and for the superseded versions of republished articles,
which no publication type expresses. Papers that merely have a published erratum are kept:
excluding them removed 1.08 percent of the corpus and caused 22.7 percent of external-truth
recall misses. RefType census of the 2026 corpus (3,346,504 links): CommentIn 1,268,042,
CommentOn 1,252,824, ErratumIn 390,627, ErratumFor 247,327, UpdateOf 105,713, UpdateIn 63,967,
RetractionOf 38,128, RetractionIn 36,962, ExpressionOfConcernFor 5,933, ExpressionOfConcernIn
5,363, ReprintOf 3,280, ReprintIn 3,248, RepublishedFrom 1,698, OriginalReportIn 1,615,
RepublishedIn 1,496, SummaryForPatientsIn 1,406, AssociatedDataset 462, AssociatedPublication
442, RetractedandRepublishedIn 86, RetractedandRepublishedFrom 79.

**Baseline years.** NCBI reissues the whole baseline each December under a new file prefix.
A download directory holds one baseline year; `download_pubmed` refuses to mix years because
the skip-if-present check is keyed on file name and mixed years duplicate every record.

## Screen

**Numerics.** The released checkpoint was trained and evaluated in fp32. Under vLLM on the
2,779-row test split: fp32 F1 0.9206, positive rate 26.23 percent, 183 rows per second per
GPU; bf16 F1 0.9213, positive rate 26.27 percent, 240 rows per second. The default stays fp32
so the deployed numerics match the evaluated artefact; `--dtype bfloat16` is available.

**Probability column.** The screen writes the positive-class probability as well as the
label, so the operating point can be changed without re-running the corpus (about 30
GPU-hours).

**Training set composition.** Test-set precision does not predict corpus behaviour: every
checkpoint had an in-domain false-positive rate of 3.4 to 3.6 percent while corpus fire rates
spanned 0.51 to 19.46 percent, because test negatives are in-domain gene-disease literature
and corpus negatives are ordinary abstracts. Corpus negatives drawn from the full converted
corpus (excluding G2P-cited, curated and annotated PMIDs) were added to training; the corpus
fire rate is a release check. Sampling is stratified by decade because the false-positive
rate rose from 10.66 percent in the 1980s to 25.55 percent in the 2020s.

**Atomic shard output.** Each output shard is written to a sidecar file and renamed on
completion, so a killed job cannot leave a truncated parquet that a restart would skip.

## Gene gate

**Two sources.** PubTator3 gene annotations (bulk file, filtered to text annotations that
occur in the title and abstract) are the primary source. HGNC descriptive names complement
them for papers that name the gene product but not the symbol ("arginase" for ARG1).
Descriptive names are matched in full, never symbols or family stems: 0.57 percent of
official symbols are English words, alias symbols raise ambiguity from 0.02 to 5.02 percent,
and stripping the index from a name such as "Bardet-Biedl syndrome 1" turns it into a disease
name that matches every gene of the family. The name dictionary is restricted to panel genes.

**Compound names.** HGNC joins bifunctional enzymes with a slash and tags orthologues in
parentheses; each slash part of at least two words is indexed as well, so "UDP-N-
acetylglucosamine 2-epimerase" matches GNE. Single-word parts are not indexed: "serine"
from "serine/threonine kinase" matched every such kinase in the panel.

**Identifier resolution.** Every route resolves to an HGNC id: PubTator GeneID through HGNC
`entrez_id`, descriptive names through the HGNC record, verbatim symbols through an HGNC-only
dictionary of approved, previous and alias symbols. An alias that is another gene's approved
symbol or a MONDO disease label or exact synonym is excluded. Symbol strings had let one
gene's current symbol match another gene's former symbol (OPA1 to MED12).

**Disease lexicon.** Disease names come from MONDO terms only; MONDO's gene terms (labels such
as SMS, AR, SET) are never used. All non-obsolete MONDO terms define disease names for alias
exclusion. For the abbreviation-context rule the lexicon is restricted to terms with a
germline-mutation basis and their ancestors: a symbol that is also a disease abbreviation
(SMS, MADD, DMD) is not gene evidence in an abstract that also writes the disease's label,
unless an occurrence of the symbol is used as a gene ("DMD gene", "NF1 c.").

**Verbatim fallback.** Where PubTator3 annotated no panel gene, panel symbols are matched
verbatim (case-sensitive, word-bounded, at least three characters). A small blocklist holds
symbols that collide with codons or laboratory vocabulary; CAD, SET, STAR and KIT are admitted
unless the text shows a competing expansion.

## Adjudication

**Candidates.** Every entry of each detected gene, from every G2P panel, is a candidate; the
prompt states the actual count. Same-gene entries from other panels give a paper about a
non-developmental disorder of the gene somewhere to map other than the gene's only
developmental entry; answers are then restricted to the developmental-disorder panel and the
unrestricted answer is kept in `llm_dis_map_all_panels`.

**Contextualised threads.** Each candidate is a multi-line block built from the all-panel
export, MONDO (synonyms, definition) and HPO (phenotype term names). The "Disease Definition"
and "Phenotypes" lines are dropped at run time. Former gene symbols are kept under a label
that names them as gene symbols, minus aliases that are disease names (CADASIL for NOTCH3, RTT
for MECP2), which the model otherwise read as the entry's disease. The builder ported from the
upstream repository omits its MedGen and HPOA network lookups because their results never
reached the rendered block. The order of the `Disease Synonyms` line in the released JSON came
from a set iteration and is not reproducible; the builder sorts them.

**Prompt rendering.** Placeholders are substituted directly rather than with `str.format`,
because the template contains braces. A row with no candidates is not sent to the model:
a prompt without candidates yields NO MATCH indistinguishable from a real negative.

**Answer parsing.** The text after the last `ANSWER:` marker is parsed, which tolerates
reasoning traces that quote the schema; G2P ids are extracted by pattern so decorations do
not turn a correct answer into a hallucination. Ids outside the offered set are flagged, not
removed; the clean stage drops them.

**Engine settings.** Prefix caching is on because the rubric is a fixed prefix of about
5,000 characters. Generation runs as one call per checkpoint window so vLLM can batch
continuously. Prompts longer than the model context are skipped per row rather than
aborting the batch. Work is striped across workers by row so every worker has work whatever
the file count.

**Knowledge cutoff.** Precision is compared for records published before and after the
adjudication model's training data cutoff to check that memorised literature does not
inflate the result.

## Clean stage

A mapping is kept when the id exists in the panel and was among the candidates offered for
that abstract. The earlier post-hoc gene-mention re-test used PubTator alone and overrode the
gate: on the 2026 corpus run it dropped 9,006 mappings of which 8,989 were gate-admitted
candidates. NO MATCH abstracts are written separately because some are gene-disease
relationships not yet in G2P.

## Evaluation

**Scoring.** Every view treats the answer as a set. End-to-end exact match over every fixture
abstract is the headline; per-id micro precision and recall, pair-level scoring over the
offered pairs and per-abstract exact match are reported alongside. Two X-linked entries of one
gene with the same disease name differing only in heterozygous versus hemizygous allelic
requirement are scored as one entry.

**External recall denominators.** Recall against curated literature is reported over all
truth PMIDs and over screen-positive ones. A large share of the external truth cannot be
curated from the abstract (reviews, pre-molecular reports, papers naming no gene), so recall
on that denominator has a ceiling below one.

**Paired comparisons.** Two screens are compared with an exact McNemar test on per-item
outcomes and a bootstrap interval on the F1 difference; three or more with Cochran's Q and
Holm-corrected pairs. Aggregate F1 differences of a few thousandths are within seed spread.

**Precision audit.** Precision of the released map is measured by a blinded, stratified
manual audit (strata: recency, disease volume, gene multiplicity) with Wilson intervals, and
inter-annotator agreement by Cohen's kappa on an overlap subset.

## Datamap figure

The overview map embeds each mapped paper's title and abstract with the MedEmbed-large-v0.1
bi-encoder, reduces the embeddings with UMAP and clusters them with HDBSCAN, then labels each
cluster with the lowest shared Mondo ancestor of its entries. The embedding serves only the
figure. A layout from the screen model's own encoder (mean-pooled LitDD BERT states) was tried:
it left 78 percent of papers unclustered at the default settings and 66 percent at looser
ones, against 35 percent for the bi-encoder, because a classifier fine-tuned for one binary
decision does not preserve topical similarity between papers.
