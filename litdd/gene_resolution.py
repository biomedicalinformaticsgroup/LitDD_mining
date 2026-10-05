"""Gene resolution by HGNC identifier for the gene gate, and the MONDO disease lexicon it uses.

Reads ``hgnc_complete_set.txt``, the G2P export CSV and a MONDO OBO release. Computes, per
abstract, the G2P entries whose gene the title and abstract name, with every route resolved to
an HGNC ID and joined to G2P on its ``hgnc id`` column:

- PubTator3 route: a GeneID resolves to an HGNC ID through the HGNC ``entrez_id`` column. The
  gene counts when one of its PubTator mention strings occurs in the text and is not a disease
  name in that text, or when its approved symbol is written case-sensitively and is not a
  disease abbreviation in context.
- Name route: HGNC descriptive names (approved, alias and previous) of panel genes, with names
  that are MONDO disease strings removed from the dictionary.
- Verbatim route, applied only where the PubTator3 route found no panel gene: a symbol
  dictionary of each panel gene's approved symbol plus its HGNC previous and alias symbols,
  excluding an alias that is the approved symbol of another HGNC gene or a MONDO disease string.
  An alias shared by several panel genes admits each of them.

Disease strings are the label and EXACT synonyms of non-obsolete ``MONDO:`` terms; MONDO's gene
terms (identifiers.org/hgnc ids) are not used. ``DiseaseLexicon.from_obo`` builds the lexicon
of every MONDO disease term, which judges aliases, names and PubTator mention strings;
``DiseaseLexicon.germline_lineage`` derives the subset of terms with a ``has material basis in
germline mutation in`` relationship to an HGNC gene plus their ``is_a`` ancestors, used by the
disease-abbreviation context rule for approved symbols.

Context rule: a symbol that is also a MONDO abbreviation (SMS for Smith-Magenis syndrome, DMD
for Duchenne muscular dystrophy) is not gene evidence in an abstract that also writes the MONDO
label of that disease (or the label without a trailing syndrome/disease/disorder), unless one
occurrence of the symbol is used as a gene ("DMD gene", "SMS variants", "a mutation in NF1").
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from litdd.genes import (
    GeneNameMatcher,
    g2p_hgnc_ids,
    mention_in_text,
    normalise_phrase,
    split_pipe,
    symbol_admissible,
    symbol_written,
    token_pattern,
    uppercase_tokens,
)

TRIM_SUFFIXES = ("syndrome", "disease", "disorder")
MIN_TRIMMED_LEN = 6
# An occurrence of an abbreviation followed by one of these is used as a gene.
GENE_USE_AFTER = re.compile(
    r"[\s-]*(?:genes?|proteins?|mRNAs?|cDNAs?|transcripts?|promoters?|exons?|introns?|alleles?|"
    r"variants?|mutations?|polypeptides?|deletions?|duplications?|expression|knock-?outs?|"
    r"knock-?downs?|loci|locus|coding|encoding|products?|sequencing|analysis|nonsense|missense|"
    r"frameshift|splice|splicing|truncating|pathogenic|germline|somatic|de novo|heterozygous|"
    r"homozygous|hemizygous|microdeletions?|haploinsufficiency|copy[\s-]number)(?![A-Za-z])"
    r"|\s*[cp]\.\s*[\d*\-(A-Z]")
# ... or preceded by a variant noun and a preposition: "a missense mutation in NF1".
GENE_USE_BEFORE = re.compile(
    r"(?:mutations?|variants?|variations?|alterations?|deletions?|duplications?|defects?|"
    r"changes?|lesions?|expression|sequencing|analysis|screening|testing)\s+"
    r"(?:in|of|within|at|on)\s+(?:the\s+|both\s+)?$", re.I)

GERMLINE_BASIS_PREFIXES = ("relationship: has_material_basis_in_germline_mutation_in ",
                           "intersection_of: has_material_basis_in_germline_mutation_in ")


@dataclass
class MondoTerms:
    """Non-obsolete ``MONDO:`` terms of one OBO release."""

    names: dict[str, tuple[str, list[str]]]        # MONDO id -> (label, [label, EXACT synonyms])
    parents: dict[str, set[str]]                   # MONDO id -> is_a MONDO ids
    basis_genes: dict[str, set[str]]               # MONDO id -> HGNC ids (germline basis)

    @classmethod
    def parse(cls, path: str) -> MondoTerms:
        """Parse an OBO file, keeping ``[Term]`` stanzas whose id starts with ``MONDO:`` and that
        are not obsolete. Synonyms are kept only when tagged EXACT."""
        names: dict[str, tuple[str, list[str]]] = {}
        parents: dict[str, set[str]] = defaultdict(set)
        basis: dict[str, set[str]] = {}
        tid, label, syns, obsolete, genes, isa = None, None, [], False, set(), set()

        def flush() -> None:
            if not tid or not tid.startswith("MONDO:") or obsolete:
                return
            names[tid] = (label or "", list(syns))
            parents[tid] |= isa
            if genes:
                basis[tid] = set(genes)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("[Term]") or line.startswith("[Typedef]"):
                    flush()
                    tid, label, syns, obsolete, genes, isa = None, None, [], False, set(), set()
                elif line.startswith("is_a: MONDO:"):
                    isa.add(line.split()[1])
                elif line.startswith(GERMLINE_BASIS_PREFIXES):
                    target = line.split()[2]
                    if "identifiers.org/hgnc/" in target:
                        genes.add("HGNC:" + target.rsplit("/", 1)[1])
                elif line.startswith("id: "):
                    tid = line[4:].strip()
                elif line.startswith("name: "):
                    label = line[6:].strip()
                    syns.append(label)
                elif line.startswith("synonym: ") and " EXACT" in line:
                    syns.append(line.split('"')[1].strip())
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
        flush()
        return cls(names, dict(parents), basis)

    def germline_lineage_ids(self) -> set[str]:
        """Terms with a germline-basis gene, plus every ``is_a`` ancestor of those terms."""
        keep = set(self.basis_genes) & set(self.names)
        stack = list(keep)
        while stack:
            for par in self.parents.get(stack.pop(), ()):
                if par in self.names and par not in keep:
                    keep.add(par)
                    stack.append(par)
        return keep


@dataclass
class DiseaseLexicon:
    """Disease strings (label and EXACT synonyms) of a set of MONDO terms, in lookup forms."""

    exact: dict[str, set[str]]                   # exact string -> MONDO ids
    normed: dict[str, set[str]]                  # normalised string -> MONDO ids
    abbrev_upper: dict[str, set[str]]            # upper-cased single-token string -> MONDO ids
    labels: dict[str, str]                       # MONDO id -> label
    label_names: dict[str, set[str]]             # MONDO id -> normalised label (and trimmed form)
    basis_genes: dict[str, set[str]]             # MONDO id -> HGNC ids, for every parsed term
    terms: MondoTerms = field(repr=False)        # the parse this lexicon was built from

    @classmethod
    def from_obo(cls, path: str) -> DiseaseLexicon:
        """Lexicon of every non-obsolete MONDO term in the OBO file at ``path``."""
        terms = MondoTerms.parse(path)
        return cls.from_terms(terms, set(terms.names))

    @classmethod
    def from_terms(cls, terms: MondoTerms, keep: Iterable[str]) -> DiseaseLexicon:
        """Lexicon of the MONDO ids in ``keep`` drawn from an existing parse."""
        exact: dict[str, set[str]] = defaultdict(set)
        normed: dict[str, set[str]] = defaultdict(set)
        abbrev: dict[str, set[str]] = defaultdict(set)
        labels: dict[str, str] = {}
        label_names: dict[str, set[str]] = {}
        for tid in keep:
            label, names = terms.names[tid]
            labels[tid] = label
            lk = normalise_phrase(label)
            label_names[tid] = {lk}
            toks = lk.split()
            if toks and toks[-1] in TRIM_SUFFIXES and len(" ".join(toks[:-1])) >= MIN_TRIMMED_LEN:
                label_names[tid].add(" ".join(toks[:-1]))
            for n in dict.fromkeys(names):
                exact[n].add(tid)
                normed[normalise_phrase(n)].add(tid)
                if " " not in n.strip():
                    abbrev[n.upper()].add(tid)
        return cls(dict(exact), dict(normed), dict(abbrev), labels, label_names,
                   terms.basis_genes, terms)

    def germline_lineage(self) -> DiseaseLexicon:
        """The sub-lexicon of terms with a germline-basis gene and their ``is_a`` ancestors."""
        return self.from_terms(self.terms, self.terms.germline_lineage_ids())

    def is_disease_symbol(self, s: str) -> bool:
        """``s`` is exactly a disease label or EXACT synonym."""
        return s in self.exact

    def is_disease_name(self, s: str) -> bool:
        """``s`` normalises to a disease label or EXACT synonym."""
        return normalise_phrase(s) in self.normed

    def is_disease_here(self, s: str, text: str, padded_norm_text: str) -> bool:
        """A multi-word string is a disease when it is a disease name; a single token is a
        disease only where ``disease_context`` finds its disease's label written in ``text``."""
        if len(normalise_phrase(s).split()) > 1:
            return self.is_disease_name(s)
        return bool(self.disease_context(s, text, padded_norm_text))

    def disease_context(self, abbrev: str, text: str, padded_norm_text: str) -> list[str]:
        """MONDO ids of diseases abbreviated ``abbrev`` whose label is written in ``text``.

        ``padded_norm_text`` is the normalised text with a space at each end. The result is
        empty when ``abbrev`` is not a disease abbreviation, when no label is present, or when
        any occurrence of ``abbrev`` in ``text`` is used as a gene (``GENE_USE_AFTER`` /
        ``GENE_USE_BEFORE``).
        """
        ids = self.abbrev_upper.get(abbrev.upper())
        if not ids:
            return []
        hit = [i for i in ids
               if any(f" {n} " in padded_norm_text for n in self.label_names.get(i, ()))]
        if not hit:
            return []
        for m in token_pattern(abbrev, ignore_case=True).finditer(text):
            if GENE_USE_AFTER.match(text, m.end()) or \
                    GENE_USE_BEFORE.search(text[max(0, m.start() - 60):m.start()]):
                return []
        return hit


@dataclass
class HgncPanel:
    """Panel genes keyed by HGNC ID, with the symbol and name dictionaries that find them."""

    records: dict[str, dict[str, str]]           # hgnc_id -> HGNC row (all loci)
    entrez_to_hgnc: dict[str, str]               # NCBI GeneID -> hgnc_id
    entries: dict[str, list[str]]                # panel hgnc_id -> [g2p id]
    symbol_of: dict[str, str]                    # panel hgnc_id -> approved symbol
    verbatim: dict[str, set[str]]                # symbol string -> panel hgnc ids
    matcher: GeneNameMatcher                     # descriptive names, disease names removed
    symbol_audit: list[dict]                     # every approved/previous/alias symbol considered
    name_audit: list[dict]                       # every descriptive name removed as a disease
    unresolved_g2p: list[tuple[str, str]]        # (g2p id, hgnc id) not found in HGNC

    @classmethod
    def from_files(cls, hgnc_path: str, g2p_csv: str, lexicon: DiseaseLexicon) -> HgncPanel:
        """Build the panel from ``hgnc_complete_set.txt`` and a G2P export.

        The verbatim dictionary holds each panel gene's approved symbol and those of its HGNC
        previous and alias symbols that are neither another HGNC gene's approved symbol nor a
        disease string of ``lexicon``. The name dictionary drops approved, alias and previous
        names that are disease names of ``lexicon``. Both decisions are recorded in
        ``symbol_audit`` and ``name_audit``.
        """
        records: dict[str, dict[str, str]] = {}
        entrez: dict[str, str] = {}
        approved: dict[str, str] = {}
        with open(hgnc_path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                h = r["hgnc_id"]
                records[h] = r
                approved[r["symbol"]] = h
                if r.get("entrez_id"):
                    entrez[r["entrez_id"].strip()] = h

        entries: dict[str, list[str]] = defaultdict(list)
        unresolved: list[tuple[str, str]] = []
        for gid, raw in g2p_hgnc_ids(g2p_csv).items():
            h = raw if raw.startswith("HGNC:") else f"HGNC:{raw}"
            if h not in records:
                unresolved.append((gid, raw))
                continue
            entries[h].append(gid)
        symbol_of = {h: records[h]["symbol"] for h in entries}

        def basis_symbols(mids: list[str]) -> list[str]:
            return sorted({records[g]["symbol"] for m in mids
                           for g in lexicon.basis_genes.get(m, ()) if g in records})

        verbatim: dict[str, set[str]] = defaultdict(set)
        audit: list[dict] = []
        for h, sym in symbol_of.items():
            verbatim[sym].add(h)
            mids = sorted(lexicon.exact.get(sym, ()))
            audit.append(dict(hgnc_id=h, gene=sym, symbol=sym, source="approved_symbol",
                              kept=True,
                              reasons="approved_symbol_is_disease_abbreviation(context rule)" if mids else "",
                              mondo_ids=";".join(mids),
                              mondo_labels=";".join(lexicon.labels[m] for m in mids),
                              disease_basis_genes=";".join(basis_symbols(mids)),
                              disease_caused_by_this_gene=any(h in lexicon.basis_genes.get(m, ()) for m in mids)))
            r = records[h]
            for src, field_ in (("hgnc_previous_symbol", "prev_symbol"),
                                ("hgnc_alias_symbol", "alias_symbol")):
                for s in split_pipe(r.get(field_)):
                    reasons = []
                    if s in approved and approved[s] != h:
                        reasons.append("approved_symbol_of_other_gene")
                    if lexicon.is_disease_symbol(s):
                        reasons.append("disease_name")
                    if not reasons:
                        verbatim[s].add(h)
                    mids = sorted(lexicon.exact.get(s, ()))
                    basis = basis_symbols(mids)
                    audit.append(dict(hgnc_id=h, gene=sym, symbol=s, source=src,
                                      kept=not reasons, reasons=";".join(reasons),
                                      mondo_ids=";".join(mids),
                                      mondo_labels=";".join(lexicon.labels[m] for m in mids),
                                      disease_basis_genes=";".join(basis),
                                      disease_caused_by_this_gene=sym in basis))

        matcher = GeneNameMatcher.from_hgnc(hgnc_path, set(symbol_of.values()))
        name_audit: list[dict] = []
        fields = (("name", "hgnc_approved_name"), ("alias_name", "hgnc_alias_name"),
                  ("prev_name", "hgnc_previous_name"))
        for h, sym in symbol_of.items():
            r = records[h]
            for field_, src in fields:
                vals = [r.get(field_, "").strip().strip('"')] if field_ == "name" else split_pipe(r.get(field_))
                for v in vals:
                    if not v or not lexicon.is_disease_name(v):
                        continue
                    # Remove every indexed variant of this disease-named entry for this gene.
                    removed = 0
                    for var in GeneNameMatcher._name_variants(v):
                        key = normalise_phrase(var)
                        syms = matcher.name_to_symbols.get(key)
                        if syms and sym in syms:
                            syms.discard(sym)
                            removed += 1
                            if not syms:
                                del matcher.name_to_symbols[key]
                    mids = sorted(lexicon.normed.get(normalise_phrase(v), ()))
                    name_audit.append(dict(hgnc_id=h, gene=sym, name=v, source=src,
                                           unindexed_keys=removed, mondo_ids=";".join(mids),
                                           mondo_labels=";".join(lexicon.labels[m] for m in mids),
                                           disease_caused_by_this_gene=any(
                                               h in lexicon.basis_genes.get(m, ()) for m in mids)))
        return cls(records, entrez, dict(entries), symbol_of, dict(verbatim), matcher, audit,
                   name_audit, unresolved)


def resolve_candidates(
    rows: Iterable[tuple[str, str | None]],
    pubtator: dict[str, dict[str, set[str]]],
    panel: HgncPanel,
    lexicon: DiseaseLexicon,
    context_lexicon: DiseaseLexicon,
) -> tuple[list[list[str]], list[list[str]], dict]:
    """Candidate G2P ids and their provenance for each (pmid, tiab) row.

    ``pubtator`` is ``load_pubtator_gene_ids`` output. ``lexicon`` judges PubTator mention
    strings; ``context_lexicon`` supplies the disease-abbreviation context rule for approved
    symbols. Returns parallel lists of sorted candidate ids and of sources (``symbol_match``,
    ``name_match``, ``symbol_fallback``; a symbol match takes precedence, then a name match), and
    a report dict with the panel audit summary and corpus counts.
    """
    approved_to_h = {s: h for h, s in panel.symbol_of.items()}
    stats: Counter = Counter()
    ctx_symbols: Counter = Counter()
    dropped_mentions: Counter = Counter()
    cand_col: list[list[str]] = []
    src_col: list[list[str]] = []
    for pmid, tiab in rows:
        tiab = tiab or ""
        lowered = tiab.lower()
        padded = f" {normalise_phrase(tiab)} "

        # PubTator3 route: GeneID -> HGNC id; the gene counts on a non-disease mention in the
        # text, or on its approved symbol written and not a disease abbreviation in context.
        by_symbol: set[str] = set()
        for eid, mentions in pubtator.get(pmid, {}).items():
            h = panel.entrez_to_hgnc.get(eid)
            if h not in panel.entries:
                continue
            sym = panel.symbol_of[h]
            kept = set()
            for m in mentions:
                if m.upper() == sym.upper():
                    is_disease = bool(context_lexicon.disease_context(m, tiab, padded))
                else:
                    is_disease = lexicon.is_disease_here(m, tiab, padded)
                if is_disease:
                    dropped_mentions[m] += 1
                else:
                    kept.add(m)
            if kept and mention_in_text(tiab, kept, sym):
                by_symbol.add(h)
                continue
            if symbol_written(sym, tiab, lowered):
                if context_lexicon.disease_context(sym, tiab, padded):
                    ctx_symbols[sym] += 1
                    continue
                by_symbol.add(h)
                stats["pubtator_gene_admitted_by_official_symbol_only"] += 1

        # Name route: descriptive names, minus genes the PubTator route found.
        by_name: set[str] = set()
        for s in panel.matcher.find(tiab):
            if s in approved_to_h:
                by_name.add(approved_to_h[s])
        by_name -= by_symbol

        # Verbatim route, only where PubTator verified no panel gene in the text.
        by_verbatim: set[str] = set()
        if not by_symbol:
            for t in uppercase_tokens(tiab):
                hs = panel.verbatim.get(t)
                if not hs or not symbol_admissible(t, tiab, lowered):
                    continue
                if context_lexicon.disease_context(t, tiab, padded):
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

    audit = Counter((a["source"], a["reasons"] or "kept") for a in panel.symbol_audit)
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
    return cand_col, src_col, report
