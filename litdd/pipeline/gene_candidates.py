#!/usr/bin/env python3
"""Restrict each BERT-positive abstract to the G2P entries whose gene it mentions.

Runs between ``build_bert_positives.py`` and ``crossencode.py``. Previously the gene-mention
check was the *last* gate, applied in ``final_data_clean.py`` after the cross-encoder had
scored every abstract against all ~2,861 G2P entries. Moving it here collapses the candidate
set to the handful an abstract's genes support -- G2P holds 2,861 entries across 2,552 genes at
a median of 1 entry each -- which removes ~1,300x of the pairwise scoring and makes the number
of candidates shown to the LLM data-driven rather than a fixed top-5.

Because this is now a gate, its recall bounds the whole pipeline. Measured before adopting it
(``litdd/evaluation/gene_filter_recall.py``, results in
``revision/external_recall/gene_filter_summary.csv``): on the independent curated sets it
retains 98.8% of true (paper, gene) pairs with PubTator alone and 99.3% with the HGNC
descriptive-name complement; on the labelled annotation set it costs 3.3% recall and raises
precision from 0.254 to 0.299.

Provenance is recorded per row (``symbol_match`` / ``name_match``) so the precision audit can
report the two sources separately and either can be down-weighted later without re-running.

Example
-------
    python litdd/pipeline/gene_candidates.py \\
        --input_parquet data/pubmed_bert_positive.parquet \\
        --g2p_csv data/G2P_DD_2026-06-24.csv \\
        --gene2pubtator data/gene2pubtator3 \\
        --gene_info data/human_gene_info.gz \\
        --hgnc data/reference/hgnc_complete_set.txt \\
        --out_parquet data/bert_positive_candidates.parquet
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from litdd.genes import (  # noqa: E402
    GeneNameMatcher,
    load_gene_info,
    load_pubtator_gene_ids,
    load_pubtator_genes,
    mention_in_text,
    normalise,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_parquet", required=True, help="BERT-positive parquet (pmid, tiab)")
    p.add_argument("--g2p_csv", required=True)
    p.add_argument("--gene2pubtator", required=True, help="gene2pubtator3 bulk file (.gz or TSV)")
    p.add_argument("--gene_info", required=True, help="NCBI gene_info.gz (human filter)")
    p.add_argument("--hgnc", default=None,
                   help="hgnc_complete_set.txt for descriptive-name matching. Recommended: "
                        "recovers ~42%% of residual external-truth misses.")
    p.add_argument("--out_parquet", required=True)
    p.add_argument("--family_stems", action="store_true",
                   help="Also match enzyme-family stems, so \"the two human arginase genes\" "
                        "resolves to ARG1/ARG2 even without a numeral. Off by default: full "
                        "HGNC names only. Disease-named genes (\"Bardet-Biedl syndrome 1\") "
                        "are blocklisted from forming stems either way, so enabling this "
                        "cannot make a syndrome mention match its whole gene family.")
    p.add_argument("--symbol_fallback", action="store_true",
                   help="For abstracts where PubTator3 has NO human gene annotation at all "
                        "(coverage gap: old or title-only records), match G2P gene symbols "
                        "verbatim in the text (case-sensitive, word-bounded, >=3 characters, "
                        "minus SYMBOL_FALLBACK_BLOCKLIST). Never applied when PubTator "
                        "annotated the abstract, so it cannot add ambiguity there. Measured on "
                        "the annotated test split: recovers 9 of 9 curated abstracts PubTator "
                        "left unannotated (DMD, ITPR1, NEXMIF, SCN8A, CPLANE1).")
    p.add_argument("--no_tiab_mention_check", action="store_true",
                   help="Pre-2026-09-02 behaviour: accept every gene the bulk file links to "
                        "the PMID, including database cross-references (BioGRID/gene2pubmed) "
                        "and full-text annotations that never appear in the TIAB. Measured on "
                        "the dev split this inflates 3%% of abstracts to 20-104 'genes'.")
    p.add_argument("--disease_alias_stoplist", default=None,
                   help="TSV from build_disease_alias_stoplist.py: gene aliases that are really "
                        "disease names (CADASIL->NOTCH3, RTT/'Rett syndrome'->MECP2). A gene "
                        "is then not counted as mentioned when its only evidence in the TIAB is "
                        "such an alias -- via G2P previous symbols in the fallback, PubTator "
                        "mention strings, or HGNC descriptive names. The official symbol, or any "
                        "other mention, still admits the gene.")
    p.add_argument("--resolution", choices=("symbol", "hgnc"), default="symbol",
                   help="'symbol' (the released v5 gate): PubTator GeneID -> NCBI symbol string -> "
                        "G2P gene symbol OR G2P previous symbol. 'hgnc': every route resolves to an "
                        "HGNC ID and joins G2P on its 'hgnc id' column; verbatim symbols come from "
                        "HGNC approved/previous/alias symbols with ambiguous and disease-name "
                        "aliases excluded (litdd/gene_resolution.py). Requires --hgnc and "
                        "--mondo_obo; ignores --disease_alias_stoplist.")
    p.add_argument("--mondo_obo", default=None,
                   help="MONDO OBO release (resolution 'hgnc'): source of disease names.")
    p.add_argument("--mondo_diseases", choices=("germline", "germline_lineage", "all"),
                   default="germline",
                   help="(resolution 'hgnc') which MONDO terms count as diseases: 'germline' = terms "
                        "with a 'has material basis in germline mutation in' (RO:0004003) "
                        "relationship to an HGNC gene; 'germline_lineage' = those plus their is_a "
                        "ancestors; 'all' = every non-obsolete MONDO: term. MONDO's gene terms are "
                        "never used.")
    p.add_argument("--context_diseases", choices=("germline", "germline_lineage", "all"), default=None,
                   help="(resolution 'hgnc') MONDO terms used by the disease-abbreviation CONTEXT "
                        "rule for approved symbols; default: same as --mondo_diseases, which then "
                        "only governs which aliases, names and PubTator mentions are disease names.")
    p.add_argument("--context_names", choices=("synonyms", "label"), default="synonyms",
                   help="(resolution 'hgnc') full disease names the context rule looks for: the MONDO "
                        "label and multi-word EXACT synonyms, or the label only.")
    p.add_argument("--disease_alias_policy", choices=("exclude", "context", "hybrid"), default="exclude",
                   help="(resolution 'hgnc') previous/alias symbols and PubTator mentions that are "
                        "MONDO disease acronyms: never gene evidence ('exclude'), or not gene "
                        "evidence only where the disease's full name is also written ('context'); "
                        "'hybrid' = previous/alias symbols excluded, PubTator mentions by context.")
    p.add_argument("--exclude_shared_aliases", action="store_true",
                   help="(resolution 'hgnc') also drop an HGNC previous/alias symbol that another "
                        "HGNC gene uses as a previous/alias symbol. Off: it admits every panel gene "
                        "that uses it.")
    p.add_argument("--verbatim_all_abstracts", action="store_true",
                   help="(resolution 'hgnc') run verbatim symbol matching on every abstract, not "
                        "only where no PubTator3 panel gene was verified in the TIAB.")
    p.add_argument("--no_disease_context", action="store_true",
                   help="(resolution 'hgnc', ablation) do not discount a symbol that is a MONDO "
                        "abbreviation when the disease's full name is also written.")
    p.add_argument("--keep_approved_disease_names", action="store_true",
                   help="(resolution 'hgnc', ablation) keep HGNC approved names that are MONDO "
                        "disease names (e.g. 'Bardet-Biedl syndrome 1') in the name dictionary.")
    p.add_argument("--audit_prefix", default=None,
                   help="(resolution 'hgnc') write <prefix>_symbols.tsv, <prefix>_names.tsv and "
                        "<prefix>_stats.json: every symbol/name considered, why it was excluded, "
                        "and per-rule corpus counts.")
    p.add_argument("--keep_unmatched", action="store_true",
                   help="Keep rows with no detected gene and give them the FULL panel as "
                        "candidates, instead of dropping them. Measured as unnecessary "
                        "(the gate retains 98.8%% of external-truth pairs), but retained so "
                        "the hard-gate/hybrid comparison can be re-run.")
    return p.parse_args()


# Official G2P symbols that are also English words or clinical abbreviations. Only consulted
# by --symbol_fallback, i.e. for abstracts PubTator did not annotate at all.
#
# NARROWED 2026-09-11. The list previously blocked 13 DD-panel gene symbols outright, which
# cost real papers: CAD-related uridine-responsive epileptic encephalopathy lost its entire
# mined set this way. Two facts drove the revision, both measured on the 179,747 screen-positive
# abstracts (25,842 of which PubTator annotated with nothing, the only population the fallback
# sees):
#
#   1. Matching is ALREADY uppercase-only and word-bounded, so the English-word collisions the
#      list was built for ("cat", "set", "was", "she") were never matching in the first place.
#      For 9 of the 13 DD genes, every fallback-eligible uppercase match was the gene:
#        ATM 8/8 · GAN 5/5 · SON 6/6 · SHOX 1/1 · REST 1/1 · KIT, NODAL, PIGN, PIGS 0 matches
#   2. Three symbols DO collide as uppercase acronyms, but only when a specific competing
#      expansion is present in the same abstract. Those are handled by context below rather
#      than by a blanket block.
#
# TAT stays blocked, and not because of a word collision: 4 of its 5 fallback-eligible matches
# were DNA codons ("stop codon (TAG) ... for Tyr (TAT)", "523 GAT-->TAT"), which carry exactly
# the same token boundaries as the symbol. It costs nothing to keep -- PubTator annotates TAT
# reliably and the HGNC descriptive name "tyrosine aminotransferase" covers the rest, so the
# TAT entry gained 16 papers in the released map rather than losing any to this block.
SYMBOL_FALLBACK_BLOCKLIST = frozenset({
    # Not G2P gene symbols at all -- the panel intersection already excludes them. Retained
    # so the list still bites should any of these ever be curated onto a panel.
    "CAT", "WAS", "ACHE", "MARS", "ARC", "BAD", "BID", "CAP", "COIL", "DIP", "FAT", "FLOT",
    "HIP", "IMPACT", "LAMP", "LARGE", "MICE", "OCT", "RAN", "SHE", "SPAG", "TANK", "WARS",
    "APEX", "CRISP", "MASS", "MICAL", "PALM", "SCAN", "SLIT", "TRIP", "WISP", "ARMS", "ATP",
    "DNA", "RNA", "EEG", "MRI", "CNS", "IQ", "ASD", "ADHD", "PCR", "CGH", "SNP", "CNV",
    "NGS", "WES", "WGS", "HPO", "MIM", "OMIM",
    # Non-DD G2P panels (Cancer/Eye); left blocked because this pipeline curates DD only.
    "AIP", "MAX", "MET", "TUB",
    # Collides with the DNA codon, which the token boundaries cannot separate.
    "TAT",
})

# Symbols admitted by the fallback UNLESS the text shows the match is something else. A blanket
# block on these three lost more than it saved, but they are genuinely ambiguous, so the
# ambiguity is resolved from the text rather than by refusing the symbol outright. Both rules
# below were fitted to, and are reported against, the 81 abstracts the narrowing newly admits
# across all 179,748 screen-positive records (measure_blocklist_corpus.py).
#
# Document-level: the competing expansion appears anywhere in the abstract, so no occurrence of
# the token is the gene.
AMBIGUOUS_SYMBOL_CONTEXT: dict[str, tuple[str, ...]] = {
    "CAD": ("coronary artery disease", "coronary atherosclerotic disease", "atheroscler",
            "premature cad", "computer-aided design", "computer aided design",
            "computer-aided diagnosis", "computer aided diagnosis",
            "computer-aided detection", "computer aided detection"),
    "SET": ("short exercise test",),
    "STAR": ("short telomere associated retinopathy", "short telomere-associated retinopathy",
             # "toe syndactyly, telecanthus and anogenital and renal malformations" -- CCNQ
             "star syndrome", "telecanthus"),
}

# Occurrence-level: the token is part of a longer name, or of a product name. Applied at each
# match; the symbol is admitted if ANY occurrence is clean, so an abstract saying "variants in
# the SET gene" still counts even where it later writes "SET domain". SET is why this mechanism
# exists -- "SET domain", "SET binding factor" (SBF1/SBF2) and "SET and MYND domain" (SMYD1) are
# standard protein nomenclature, and accounted for 11 of the 17 false matches that the
# document-level rule alone let through.
AMBIGUOUS_SYMBOL_FOLLOWED_BY: dict[str, re.Pattern[str]] = {
    "SET": re.compile(r"\s*(?:binding factor|domain|and MYND|\(BP1\))", re.I),
}
# Laboratory products written in caps: "cycle sequencing KIT", "SALSA MLPA KIT P245".
AMBIGUOUS_SYMBOL_PRECEDED_BY: dict[str, re.Pattern[str]] = {
    "KIT": re.compile(r"(?:sequencing|MLPA|PCR|SALSA|extraction|amplification|assay|detection|"
                      r"isolation|purification|labell?ing|hybridi[sz]ation)\s+$", re.I),
}


def _blocked_by_context(symbol: str, text: str, lowered: str) -> bool:
    """True when the text shows this token is not the gene."""
    if any(x in lowered for x in AMBIGUOUS_SYMBOL_CONTEXT.get(symbol, ())):
        return True
    after_pat = AMBIGUOUS_SYMBOL_FOLLOWED_BY.get(symbol)
    before_pat = AMBIGUOUS_SYMBOL_PRECEDED_BY.get(symbol)
    if after_pat is None and before_pat is None:
        return False
    for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", text):
        if after_pat is not None and after_pat.match(text, m.end()):
            continue
        if before_pat is not None and before_pat.search(text[max(0, m.start() - 40):m.start()]):
            continue
        return False        # this occurrence stands alone -> the symbol is admitted
    return True


def find_symbols_verbatim(text: str, symbols: set[str]) -> set[str]:
    """Panel symbols appearing verbatim (case-sensitive, word-bounded) in ``text``."""
    if not text:
        return set()
    # hyphenated symbols (NKX2-1, HLA-DRB1) are kept whole; a hyphen followed by lowercase
    # ("ITPR1-related") ends the token so the bare symbol still matches
    tokens = set(re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![A-Za-z0-9])", text))
    lowered = text.lower()
    return {t for t in tokens
            if len(t) >= 3 and t in symbols
            and t not in SYMBOL_FALLBACK_BLOCKLIST
            and not _blocked_by_context(t, text, lowered)}


def load_gene_to_g2p(path: str) -> dict[str, list[str]]:
    """gene symbol (and previous symbols) -> [g2p_id, ...]."""
    out: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gid = (row.get("g2p id") or "").strip()
            if not gid:
                continue
            syms = [(row.get("gene symbol") or "").strip()]
            syms += [s.strip() for s in (row.get("previous gene symbols") or "").split(";")]
            for s in syms:
                if s:
                    out.setdefault(s, []).append(gid)
    return out


def load_disease_alias_stoplist(path: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """-> ({gene: {ALIAS SYMBOLS upper-cased}}, {gene: {normalised alias names}})."""
    symbols: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            gene, alias = row["gene"].strip(), row["alias"].strip()
            if row["kind"] == "symbol":
                symbols.setdefault(gene, set()).add(alias.upper())
            else:
                names.setdefault(gene, set()).add(" ".join(normalise(alias)))
    return symbols, names


def _symbol_written(sym: str, text: str, lowered: str) -> bool:
    """Case-sensitive, word-bounded, >= 3 characters, not blocklisted, not declined by context."""
    return (len(sym) >= 3 and sym not in SYMBOL_FALLBACK_BLOCKLIST
            and re.search(rf"(?<![A-Za-z0-9]){re.escape(sym)}(?![A-Za-z0-9])", text) is not None
            and not _blocked_by_context(sym, text, lowered))


def run_hgnc_resolution(args, df: pl.DataFrame) -> tuple[list, list, dict]:
    """Candidate G2P ids per row with every gene resolved by HGNC ID (see litdd/gene_resolution.py).

    PubTator3 route: a GeneID resolves to an HGNC ID through HGNC entrez_id; the gene counts when
    a mention string that is not a disease name occurs in the TIAB, or its approved symbol is
    written (case-sensitive) and is not a disease abbreviation in context. Name route: HGNC
    descriptive names with disease names removed. Verbatim route (where no PubTator panel gene
    was verified, or everywhere with --verbatim_all_abstracts): the HGNC-only symbol dictionary.
    """
    import json
    from collections import Counter

    from litdd.gene_resolution import DiseaseLexicon, HgncPanel

    if not (args.hgnc and args.mondo_obo):
        raise SystemExit("--resolution hgnc requires --hgnc and --mondo_obo")
    lex = DiseaseLexicon.from_obo(args.mondo_obo, mode=args.mondo_diseases,
                                  context_names=args.context_names)
    ctx = lex if args.context_diseases in (None, args.mondo_diseases) else \
        DiseaseLexicon.from_obo(args.mondo_obo, mode=args.context_diseases,
                                context_names=args.context_names)
    panel = HgncPanel.from_files(args.hgnc, args.g2p_csv, lex,
                                 exclude_approved_disease_names=not args.keep_approved_disease_names,
                                 exclude_shared_aliases=args.exclude_shared_aliases,
                                 disease_alias_policy=("context" if args.disease_alias_policy == "context"
                                                       else "exclude"))
    audit = Counter((a["source"], a["reasons"] or "kept") for a in panel.symbol_audit)
    print(f"MONDO disease terms: {len(lex.labels):,}; panel genes by HGNC ID: {len(panel.entries):,} "
          f"({sum(map(len, panel.entries.values())):,} entries; {len(panel.unresolved_g2p)} G2P rows "
          f"with an HGNC id not in HGNC)", flush=True)
    for (src, why), n in sorted(audit.items()):
        print(f"  {src:22s} {why:60s} {n:6,}", flush=True)
    print(f"  descriptive names removed as disease names: {len(panel.name_audit):,}", flush=True)

    pmids = set(df["pmid"].cast(pl.Utf8).to_list())
    pub = load_pubtator_gene_ids(args.gene2pubtator, pmids)
    print(f"pmids with >=1 PubTator3 gene annotation: {len(pub):,}", flush=True)

    approved_to_h = {s: h for h, s in panel.symbol_of.items()}
    stats = Counter()
    ctx_symbols, dropped_mentions = Counter(), Counter()
    cand_col, src_col = [], []
    for pmid, tiab in zip(df["pmid"].cast(pl.Utf8).to_list(), df["tiab"].to_list()):
        tiab = tiab or ""
        lowered = tiab.lower()
        padded = f" {' '.join(normalise(tiab))} "
        by_symbol: set[str] = set()
        for eid, mentions in pub.get(pmid, {}).items():
            h = panel.entrez_to_hgnc.get(eid)
            if h not in panel.entries:
                continue
            sym = panel.symbol_of[h]
            kept = set()
            for m in mentions:
                if m.upper() == sym.upper():
                    is_disease = (not args.no_disease_context
                                  and bool(ctx.disease_context(m, tiab, padded)))
                elif args.disease_alias_policy in ("context", "hybrid"):
                    is_disease = lex.is_disease_here(m, tiab, padded)
                else:
                    is_disease = lex.is_disease_name(m) or lex.is_disease_symbol(m)
                if is_disease:
                    dropped_mentions[m] += 1
                else:
                    kept.add(m)
            if kept and mention_in_text(tiab, kept, sym):
                by_symbol.add(h)
                continue
            if _symbol_written(sym, tiab, lowered):
                if not args.no_disease_context and ctx.disease_context(sym, tiab, padded):
                    ctx_symbols[sym] += 1
                    continue
                by_symbol.add(h)
                stats["pubtator_gene_admitted_by_official_symbol_only"] += 1

        by_name: set[str] = set()
        for s in panel.matcher.find(tiab):
            if s in approved_to_h:
                by_name.add(approved_to_h[s])
        by_name -= by_symbol

        by_verbatim: set[str] = set()
        if args.symbol_fallback and (args.verbatim_all_abstracts or not by_symbol):
            for t in set(re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![A-Za-z0-9])",
                                    tiab)):
                hs = panel.verbatim.get(t)
                if not hs or len(t) < 3 or t in SYMBOL_FALLBACK_BLOCKLIST \
                        or _blocked_by_context(t, tiab, lowered):
                    continue
                is_approved = t in approved_to_h
                if not args.no_disease_context and ctx.disease_context(t, tiab, padded):
                    ctx_symbols[t] += 1
                    continue
                if (not is_approved and args.disease_alias_policy == "context"
                        and lex.is_disease_symbol(t) and lex.disease_context(t, tiab, padded)):
                    ctx_symbols[t] += 1
                    continue
                by_verbatim |= hs
            by_verbatim -= by_name | by_symbol

        ids: dict[str, str] = {}
        for h in by_symbol:
            for gid in panel.entries[h]:
                ids[gid] = "symbol_match"
        for h in by_name:
            for gid in panel.entries[h]:
                ids.setdefault(gid, "name_match")
        for h in by_verbatim:
            for gid in panel.entries[h]:
                ids.setdefault(gid, "symbol_fallback")
        stats["rows_symbol"] += bool(by_symbol)
        stats["rows_name"] += bool(by_name)
        stats["rows_verbatim"] += bool(by_verbatim)
        ordered = sorted(ids)
        cand_col.append(ordered)
        src_col.append([ids[g] for g in ordered])

    report = dict(
        symbol_audit=[dict(source=s, outcome=w, n=n) for (s, w), n in sorted(audit.items())],
        names_removed_as_disease=len(panel.name_audit),
        names_removed_by_source=dict(Counter(a["source"] for a in panel.name_audit)),
        unresolved_g2p=panel.unresolved_g2p,
        disease_context_abstract_symbol_pairs=sum(ctx_symbols.values()),
        disease_context_top=ctx_symbols.most_common(40),
        disease_mentions_dropped=sum(dropped_mentions.values()),
        disease_mentions_top=dropped_mentions.most_common(40),
        **stats,
    )
    if args.audit_prefix:
        os.makedirs(os.path.dirname(args.audit_prefix) or ".", exist_ok=True)
        pl.DataFrame(panel.symbol_audit).write_csv(f"{args.audit_prefix}_symbols.tsv", separator="\t")
        if panel.name_audit:
            pl.DataFrame(panel.name_audit).write_csv(f"{args.audit_prefix}_names.tsv", separator="\t")
        with open(f"{args.audit_prefix}_stats.json", "w") as f:
            json.dump(report, f, indent=1)
    return cand_col, src_col, report


def main() -> int:
    args = parse_args()

    if args.resolution == "hgnc":
        df = pl.read_parquet(args.input_parquet)
        if "pmid" not in df.columns or "tiab" not in df.columns:
            raise SystemExit("--input_parquet must have 'pmid' and 'tiab' columns")
        print(f"input rows: {df.height}  resolution: hgnc", flush=True)
        cand_col, src_col, report = run_hgnc_resolution(args, df)
        df = df.with_columns([
            pl.Series("candidate_g2p_ids", cand_col, dtype=pl.List(pl.Utf8)),
            pl.Series("candidate_sources", src_col, dtype=pl.List(pl.Utf8)),
        ])
        kept = df.filter(pl.col("candidate_g2p_ids").list.len() > 0)
        n_cand = kept["candidate_g2p_ids"].list.len()
        print(f"\nrows retained            : {kept.height} / {df.height}")
        if kept.height:
            print(f"total (tiab, candidate) pairs: {int(n_cand.sum()):,}")
        print(f"disease-abbreviation context declined {report['disease_context_abstract_symbol_pairs']:,} "
              f"(abstract, symbol); disease-name mentions dropped {report['disease_mentions_dropped']:,}")
        os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
        kept.write_parquet(args.out_parquet, compression="zstd")
        print(f"\nwrote {args.out_parquet}")
        return 0

    gene_to_g2p = load_gene_to_g2p(args.g2p_csv)
    stop_symbols: dict[str, set[str]] = {}
    stop_names: dict[str, set[str]] = {}
    if args.disease_alias_stoplist:
        stop_symbols, stop_names = load_disease_alias_stoplist(args.disease_alias_stoplist)
        # previous symbols live in gene_to_g2p as extra keys; the stop-listed ones must not be
        # matchable by the verbatim fallback (the builder never lists an official symbol)
        dropped = {a for aliases in stop_symbols.values() for a in aliases}
        n_before = len(gene_to_g2p)
        gene_to_g2p = {s: ids for s, ids in gene_to_g2p.items() if s not in dropped}
        print(f"disease-alias stop list: {sum(map(len, stop_symbols.values()))} symbols, "
              f"{sum(map(len, stop_names.values()))} names on "
              f"{len(set(stop_symbols) | set(stop_names))} genes; "
              f"{n_before - len(gene_to_g2p)} previous-symbol keys removed from the fallback",
              flush=True)
    panel_symbols = set(gene_to_g2p)
    all_g2p_ids = sorted({g for ids in gene_to_g2p.values() for g in ids})
    print(f"G2P panel: {len(all_g2p_ids)} entries over {len(panel_symbols)} symbols", flush=True)

    df = pl.read_parquet(args.input_parquet)
    if "pmid" not in df.columns or "tiab" not in df.columns:
        raise SystemExit("--input_parquet must have 'pmid' and 'tiab' columns")
    pmids = set(df["pmid"].cast(pl.Utf8).to_list())
    print(f"input rows: {df.height}  unique pmids: {len(pmids)}", flush=True)

    gene_info = load_gene_info(args.gene_info)
    print(f"human genes in gene_info: {len(gene_info)}", flush=True)
    pub_genes = load_pubtator_genes(args.gene2pubtator, pmids, gene_info,
                                    text_annotations_only=not args.no_tiab_mention_check,
                                    with_mentions=not args.no_tiab_mention_check)
    print(f"pmids with >=1 PubTator human gene: {len(pub_genes)}", flush=True)

    matcher = None
    if args.hgnc:
        matcher = GeneNameMatcher.from_hgnc(args.hgnc, panel_symbols,
                                            family_stems=args.family_stems)
        print(f"HGNC names indexed: {len(matcher.name_to_symbols)} "
              f"families: {len(matcher.family_to_symbols)}", flush=True)
        n_unindexed = 0
        for gene, keys in stop_names.items():
            for key in keys:
                syms = matcher.name_to_symbols.get(key)
                if syms and gene in syms:
                    syms.discard(gene)
                    n_unindexed += 1
                    if not syms:
                        del matcher.name_to_symbols[key]
        if stop_names:
            print(f"disease-alias stop list: {n_unindexed} disease-named HGNC names unindexed",
                  flush=True)

    def _gene_mentions(sym: str, mentions: set[str]) -> set[str]:
        """PubTator mention strings minus those that are disease names for this gene.

        When every mention was a disease alias, the official symbol is the only needle left,
        so the gene still counts if the TIAB writes the symbol itself."""
        stop_s, stop_n = stop_symbols.get(sym), stop_names.get(sym)
        if not mentions or (stop_s is None and stop_n is None):
            return mentions
        kept = {m for m in mentions
                if m.upper() not in (stop_s or ())
                and " ".join(normalise(m)) not in (stop_n or ())}
        return kept or {sym}

    cand_col: list[list[str]] = []
    src_col: list[list[str]] = []
    n_symbol = n_name = n_fallback = n_none = 0

    for pmid, tiab in zip(df["pmid"].cast(pl.Utf8).to_list(), df["tiab"].to_list()):
        if args.no_tiab_mention_check:
            by_symbol = pub_genes.get(pmid, set()) & panel_symbols
        else:
            # The bulk file's gene list covers full text and database cross-references; the
            # gate classifies a TIAB, so a gene counts only if one of its PubTator mention
            # strings actually occurs in the title+abstract.
            by_symbol = {sym for sym, men in pub_genes.get(pmid, {}).items()
                         if sym in panel_symbols
                         and mention_in_text(tiab or "", _gene_mentions(sym, men), sym)}
        by_name = (matcher.find(tiab or "") if matcher is not None else set()) - by_symbol
        # PubTator has no annotation for this PMID at all -> verbatim panel symbols in the text
        by_fallback: set[str] = set()
        if args.symbol_fallback and not by_symbol:
            by_fallback = find_symbols_verbatim(tiab or "", panel_symbols) - by_name
        if by_symbol:
            n_symbol += 1
        if by_name:
            n_name += 1
        if by_fallback:
            n_fallback += 1

        ids: dict[str, str] = {}
        for sym in by_symbol:
            for gid in gene_to_g2p.get(sym, []):
                ids[gid] = "symbol_match"
        for sym in by_name:
            for gid in gene_to_g2p.get(sym, []):
                ids.setdefault(gid, "name_match")
        for sym in by_fallback:
            for gid in gene_to_g2p.get(sym, []):
                ids.setdefault(gid, "symbol_fallback")

        if not ids:
            n_none += 1
            if args.keep_unmatched:
                ids = {gid: "fallback_full_panel" for gid in all_g2p_ids}
        ordered = sorted(ids)
        cand_col.append(ordered)
        src_col.append([ids[g] for g in ordered])

    df = df.with_columns([
        pl.Series("candidate_g2p_ids", cand_col, dtype=pl.List(pl.Utf8)),
        pl.Series("candidate_sources", src_col, dtype=pl.List(pl.Utf8)),
    ])

    kept = df.filter(pl.col("candidate_g2p_ids").list.len() > 0)
    n_cand = kept["candidate_g2p_ids"].list.len()

    print(f"\nrows with a symbol match : {n_symbol}")
    print(f"rows adding a name match : {n_name}")
    print(f"rows via symbol fallback : {n_fallback} (PubTator had no annotation for the PMID)")
    print(f"rows with no gene        : {n_none} "
          f"({'kept via full-panel fallback' if args.keep_unmatched else 'DROPPED'})")
    print(f"rows retained            : {kept.height} / {df.height} "
          f"({100 * kept.height / max(df.height, 1):.1f}%)")
    if kept.height:
        print(f"candidates per row       : mean {n_cand.mean():.2f}  median {n_cand.median():.0f} "
              f"max {n_cand.max()}")
        print(f"total (tiab, candidate) pairs to score: {int(n_cand.sum()):,} "
              f"(vs {df.height * len(all_g2p_ids):,} for the full panel)")

    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    kept.write_parquet(args.out_parquet, compression="zstd")
    print(f"\nwrote {args.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
