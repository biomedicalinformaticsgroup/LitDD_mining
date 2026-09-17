"""Identifier-based gene resolution for the gene gate, and the disease lexicon it relies on.

The released gate resolved a PubTator3 NCBI GeneID to its NCBI *symbol* and looked that string
up among G2P gene symbols and G2P "previous gene symbols". Matching strings rather than
identifiers lets one gene's current symbol hit another gene's former one (PubTator's OPA1 ->
MED12, AR -> FDXR, FLG -> FGFR1) and misses genes whose NCBI symbol differs from G2P's
(TRNL1 vs MT-TL1). Here every route resolves to an HGNC ID, and G2P entries are joined on the
G2P ``hgnc id`` column:

  PubTator3  GeneID -> HGNC (entrez_id) -> HGNC ID
  names      HGNC descriptive names of panel genes -> HGNC ID
  verbatim   a symbol dictionary built from HGNC only: each panel gene's approved symbol plus
             its HGNC previous/alias symbols, EXCLUDING an alias that is
               - the approved symbol of another HGNC gene,
               - a MONDO disease label or EXACT synonym (a disease name, e.g. RTT, SLOS), or
               - optionally, also a previous/alias symbol of another HGNC gene (ambiguous);
                 when kept, a shared alias admits every panel gene that uses it.
             G2P "previous gene symbols" are not used: 1,185 of 7,210 are not HGNC previous or
             alias symbols of the gene, and 103 are another gene's approved symbol.

Disease names are derived from MONDO disease terms only. The MONDO OBO release also carries
HGNC/NCBI gene terms whose labels are gene symbols (SMS, AR, SET); those are never used. By
``mode``: 'germline' = a MONDO term with a ``has material basis in germline mutation in``
(RO:0004003) relationship to an HGNC gene -- the monogenic diseases this pipeline curates;
'germline_lineage' = those terms plus their MONDO is_a ancestors, so grouping diseases of
genetic conditions (craniosynostosis, Rubinstein-Taybi syndrome, MODY) count too; 'all' =
every non-obsolete ``MONDO:`` term. The basis genes are kept for the audit.

Disease-abbreviation context. A symbol that is also a MONDO abbreviation (SMS = Smith-Magenis
syndrome, MADD = multiple acyl-CoA dehydrogenase deficiency, DMD = Duchenne muscular
dystrophy) is not gene evidence in an abstract that also writes that disease's full name (or
the name without a trailing "syndrome"/"disease"/"disorder", e.g. "Smith-Magenis") -- unless
one occurrence of the symbol is used as a gene ("DMD gene", "SMS variants", "NF1 c.").
Genes are then admitted only through other routes.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field

from litdd.genes import GeneNameMatcher, normalise

TRIM_SUFFIXES = ("syndrome", "disease", "disorder")
MIN_TRIMMED_LEN = 6
# an occurrence of the abbreviation used as a gene, which keeps it as gene evidence
GENE_USE_AFTER = re.compile(
    r"[\s-]*(?:genes?|proteins?|mRNAs?|cDNAs?|transcripts?|promoters?|exons?|introns?|alleles?|"
    r"variants?|mutations?|polypeptides?|deletions?|duplications?|expression|knock-?outs?|"
    r"knock-?downs?|loci|locus|coding|encoding|products?|sequencing|analysis|nonsense|missense|"
    r"frameshift|splice|splicing|truncating|pathogenic|germline|somatic|de novo|heterozygous|"
    r"homozygous|hemizygous|microdeletions?|haploinsufficiency|copy[\s-]number)(?![A-Za-z])"
    r"|\s*[cp]\.\s*[\d*\-(A-Z]")
# ... or preceded by a variant noun and a preposition: "a missense mutation in NF1"
GENE_USE_BEFORE = re.compile(
    r"(?:mutations?|variants?|variations?|alterations?|deletions?|duplications?|defects?|"
    r"changes?|lesions?|expression|sequencing|analysis|screening|testing)\s+"
    r"(?:in|of|within|at|on)\s+(?:the\s+|both\s+)?$", re.I)


def _split(v: str | None) -> list[str]:
    return [x.strip().strip('"') for x in (v or "").strip('"').split("|") if x.strip().strip('"')]


def _norm(s: str) -> str:
    return " ".join(normalise(s))


@dataclass
class DiseaseLexicon:
    """MONDO disease strings (label + EXACT synonyms), MONDO: terms only."""

    exact: dict[str, set[str]]                   # exact string -> MONDO ids
    normed: dict[str, set[str]]                  # normalised string -> MONDO ids
    abbrev_upper: dict[str, set[str]]            # upper-cased single-token string -> MONDO ids
    long_names: dict[str, set[str]]              # MONDO id -> normalised multi-word names (+trims)
    labels: dict[str, str] = field(default_factory=dict)
    label_names: dict[str, set[str]] = field(default_factory=dict)   # MONDO id -> norm label (+trim)
    context_names: str = "synonyms"
    basis_genes: dict[str, set[str]] = field(default_factory=dict)   # MONDO id -> HGNC ids (RO:0004003)

    @classmethod
    def from_obo(cls, path: str, mode: str = "germline", context_names: str = "synonyms") -> "DiseaseLexicon":
        """``context_names``: which full names the context rule looks for -- 'synonyms' = the label
        and every multi-word EXACT synonym; 'label' = the MONDO label only. Both also accept the
        name without a trailing syndrome/disease/disorder."""
        if mode not in ("germline", "germline_lineage", "all"):
            raise ValueError(mode)
        exact, normed, abbrev = defaultdict(set), defaultdict(set), defaultdict(set)
        long_names, labels, basis, label_names = defaultdict(set), {}, {}, {}
        terms: dict[str, tuple[str, list[str]]] = {}
        parents: dict[str, set[str]] = defaultdict(set)
        tid, label, names, obsolete, genes, isa = None, None, [], False, set(), set()

        def flush():
            if not tid or not tid.startswith("MONDO:") or obsolete:
                return
            terms[tid] = (label or "", list(names))
            parents[tid] |= isa
            if genes:
                basis[tid] = set(genes)

        def add(tid, label, names):
            labels[tid] = label
            lk = _norm(label)
            label_names[tid] = {lk}
            if lk.split() and lk.split()[-1] in TRIM_SUFFIXES and len(" ".join(lk.split()[:-1])) >= MIN_TRIMMED_LEN:
                label_names[tid].add(" ".join(lk.split()[:-1]))
            for n in dict.fromkeys(names):
                exact[n].add(tid)
                normed[_norm(n)].add(tid)
                if " " not in n.strip():
                    abbrev[n.upper()].add(tid)
                key = _norm(n)
                if len(key.split()) >= 2:
                    long_names[tid].add(key)
                    toks = key.split()
                    if toks[-1] in TRIM_SUFFIXES:
                        trimmed = " ".join(toks[:-1])
                        if len(trimmed) >= MIN_TRIMMED_LEN:
                            long_names[tid].add(trimmed)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("[Term]") or line.startswith("[Typedef]"):
                    flush()
                    tid, label, names, obsolete, genes, isa = None, None, [], False, set(), set()
                elif line.startswith("is_a: MONDO:"):
                    isa.add(line.split()[1])
                elif (line.startswith("relationship: has_material_basis_in_germline_mutation_in ")
                      or line.startswith("intersection_of: has_material_basis_in_germline_mutation_in ")):
                    target = line.split()[2]
                    if "identifiers.org/hgnc/" in target:
                        genes.add("HGNC:" + target.rsplit("/", 1)[1])
                elif line.startswith("id: "):
                    tid = line[4:].strip()
                elif line.startswith("name: "):
                    label = line[6:].strip()
                    names.append(label)
                elif line.startswith("synonym: ") and " EXACT" in line:
                    names.append(line.split('"')[1].strip())
                elif line.startswith("is_obsolete: true"):
                    obsolete = True
        flush()
        if mode == "all":
            keep = set(terms)
        else:
            keep = set(basis) & set(terms)
            if mode == "germline_lineage":
                stack = list(keep)
                while stack:
                    for par in parents.get(stack.pop(), ()):
                        if par in terms and par not in keep:
                            keep.add(par)
                            stack.append(par)
        for t in keep:
            add(t, *terms[t])
        return cls(dict(exact), dict(normed), dict(abbrev), dict(long_names), labels, label_names,
                   context_names, basis)

    def is_disease_symbol(self, s: str) -> bool:
        return s in self.exact

    def is_disease_name(self, s: str) -> bool:
        return _norm(s) in self.normed

    def is_disease_here(self, s: str, text: str, padded_norm_text: str) -> bool:
        """Context policy: a multi-word disease name is always a disease; a one-token acronym is
        a disease only where its disease's full name is also written (``disease_context``)."""
        if len(_norm(s).split()) > 1:
            return self.is_disease_name(s)
        return bool(self.disease_context(s, text, padded_norm_text))

    def disease_context(self, abbrev: str, text: str, padded_norm_text: str) -> list[str]:
        """MONDO ids whose full name is written in ``text`` next to ``abbrev`` used as a disease.

        Empty when ``abbrev`` is not a MONDO abbreviation, no full name is present, or any
        occurrence of the abbreviation is used as a gene."""
        ids = self.abbrev_upper.get(abbrev.upper())
        if not ids:
            return []
        names = self.label_names if self.context_names == "label" else self.long_names
        hit = [i for i in ids
               if any(f" {n} " in padded_norm_text for n in names.get(i, ()))]
        if not hit:
            return []
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(abbrev) + r"(?![A-Za-z0-9])", text,
                             flags=re.I):
            if GENE_USE_AFTER.match(text, m.end()) or \
                    GENE_USE_BEFORE.search(text[max(0, m.start() - 60):m.start()]):
                return []
        return hit


@dataclass
class HgncPanel:
    """Panel genes keyed by HGNC ID, and the HGNC-only symbol dictionary for verbatim matching."""

    records: dict[str, dict]                     # hgnc_id -> HGNC row (all loci)
    entrez_to_hgnc: dict[str, str]
    entries: dict[str, list[str]]                # panel hgnc_id -> [g2p id]
    symbol_of: dict[str, str]                    # panel hgnc_id -> approved symbol
    verbatim: dict[str, set[str]]                # symbol string -> panel hgnc ids
    matcher: GeneNameMatcher
    symbol_audit: list[dict]                     # every approved/previous/alias symbol considered
    name_audit: list[dict]                       # every descriptive name removed as a disease
    unresolved_g2p: list[tuple[str, str]]        # (g2p id, hgnc id) not found in HGNC

    @classmethod
    def from_files(cls, hgnc_path: str, g2p_csv: str, lexicon: DiseaseLexicon,
                   exclude_approved_disease_names: bool = True,
                   exclude_shared_aliases: bool = False,
                   disease_alias_policy: str = "exclude") -> "HgncPanel":
        """``disease_alias_policy``: 'exclude' = a previous/alias symbol that is a MONDO disease
        string is never matched; 'context' = it is matched, but not where the disease's full name is
        also written (the caller applies the context rule)."""
        records, entrez, approved = {}, {}, {}
        owners = defaultdict(set)                # previous/alias symbol -> hgnc ids using it
        with open(hgnc_path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                h = r["hgnc_id"]
                records[h] = r
                approved[r["symbol"]] = h
                if r.get("entrez_id"):
                    entrez[r["entrez_id"].strip()] = h
                for s in _split(r.get("prev_symbol")) + _split(r.get("alias_symbol")):
                    owners[s].add(h)

        entries: dict[str, list[str]] = defaultdict(list)
        unresolved = []
        with open(g2p_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gid = (row.get("g2p id") or "").strip()
                raw = (row.get("hgnc id") or "").strip()
                if not gid:
                    continue
                h = raw if raw.startswith("HGNC:") else f"HGNC:{raw}"
                if h not in records:
                    unresolved.append((gid, raw))
                    continue
                entries[h].append(gid)
        symbol_of = {h: records[h]["symbol"] for h in entries}

        verbatim: dict[str, set[str]] = defaultdict(set)
        audit = []
        for h, sym in symbol_of.items():
            verbatim[sym].add(h)
            mids = sorted(lexicon.exact.get(sym, ()))
            audit.append(dict(hgnc_id=h, gene=sym, symbol=sym, source="approved_symbol",
                              kept=True,
                              reasons="approved_symbol_is_disease_abbreviation(context rule)" if mids else "",
                              mondo_ids=";".join(mids),
                              mondo_labels=";".join(lexicon.labels[m] for m in mids),
                              disease_basis_genes=";".join(sorted({records[g]["symbol"] for m in mids
                                                                   for g in lexicon.basis_genes.get(m, ())
                                                                   if g in records})),
                              disease_caused_by_this_gene=any(h in lexicon.basis_genes.get(m, ()) for m in mids)))
            r = records[h]
            for src, field_ in (("hgnc_previous_symbol", "prev_symbol"),
                                ("hgnc_alias_symbol", "alias_symbol")):
                for s in _split(r.get(field_)):
                    reasons = []
                    if s in approved and approved[s] != h:
                        reasons.append("approved_symbol_of_other_gene")
                    if lexicon.is_disease_symbol(s):
                        reasons.append("disease_name" if disease_alias_policy == "exclude"
                                       else "disease_acronym_context_rule")
                    if exclude_shared_aliases and owners[s] - {h}:
                        reasons.append("shared_with_other_gene")
                    excluded = [x for x in reasons if x != "disease_acronym_context_rule"]
                    if not excluded:
                        verbatim[s].add(h)
                    mids = sorted(lexicon.exact.get(s, ()))
                    basis = sorted({records[g]["symbol"] for m in mids
                                    for g in lexicon.basis_genes.get(m, ()) if g in records})
                    audit.append(dict(hgnc_id=h, gene=sym, symbol=s, source=src,
                                      kept=not excluded, reasons=";".join(reasons),
                                      mondo_ids=";".join(mids),
                                      mondo_labels=";".join(lexicon.labels[m] for m in mids),
                                      disease_basis_genes=";".join(basis),
                                      disease_caused_by_this_gene=sym in basis))

        matcher = GeneNameMatcher.from_hgnc(hgnc_path, set(symbol_of.values()))
        name_audit = []
        fields = [("alias_name", "hgnc_alias_name"), ("prev_name", "hgnc_previous_name")]
        if exclude_approved_disease_names:
            fields.insert(0, ("name", "hgnc_approved_name"))
        for h, sym in symbol_of.items():
            r = records[h]
            for field_, src in fields:
                vals = [r.get(field_, "").strip().strip('"')] if field_ == "name" else _split(r.get(field_))
                for v in vals:
                    if not v or not lexicon.is_disease_name(v):
                        continue
                    removed = 0
                    for var in GeneNameMatcher._name_variants(v):
                        key = _norm(var)
                        syms = matcher.name_to_symbols.get(key)
                        if syms and sym in syms:
                            syms.discard(sym)
                            removed += 1
                            if not syms:
                                del matcher.name_to_symbols[key]
                    mids = sorted(lexicon.normed.get(_norm(v), ()))
                    name_audit.append(dict(hgnc_id=h, gene=sym, name=v, source=src,
                                           unindexed_keys=removed, mondo_ids=";".join(mids),
                                           mondo_labels=";".join(lexicon.labels[m] for m in mids),
                                           disease_caused_by_this_gene=any(
                                               h in lexicon.basis_genes.get(m, ()) for m in mids)))
        return cls(records, entrez, dict(entries), symbol_of, dict(verbatim), matcher, audit,
                   name_audit, unresolved)

    def mention_is_disease(self, mention: str, h: str, lexicon: DiseaseLexicon, text: str,
                           padded_norm_text: str) -> bool:
        """A PubTator mention string that names a disease rather than gene ``h``."""
        if mention.upper() == self.symbol_of[h].upper():
            return bool(lexicon.disease_context(mention, text, padded_norm_text))
        return lexicon.is_disease_name(mention) or lexicon.is_disease_symbol(mention)
