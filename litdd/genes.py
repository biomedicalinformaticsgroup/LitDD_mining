"""Gene-data loaders and text-matching primitives shared by the gene gate and the cleaner.

Reads: the G2P export CSV (``g2p id``, ``hgnc id``), NCBI ``gene_info`` (GeneID to symbol,
human rows only), the ``gene2pubtator3`` bulk file (PMID, GeneID, mention strings, resource)
and ``hgnc_complete_set.txt`` (descriptive gene names).

Provides:

- ``normalise`` / ``normalise_phrase`` / ``split_pipe``: string normalisation used by every
  dictionary built here and in ``litdd.gene_resolution``.
- ``load_pubtator_genes`` (symbols, every resource) and ``load_pubtator_gene_ids`` (GeneIDs with
  mention strings, PubTator3 text annotations only).
- ``mention_in_text`` and the symbol-matching primitives (``uppercase_tokens``,
  ``symbol_admissible``, ``symbol_written``): case-sensitive, word-bounded symbol matching with
  a blocklist and context rules for symbols that are also acronyms or product names.
- ``GeneNameMatcher``: n-gram lookup of HGNC descriptive names (full names only) restricted to
  the panel genes.
"""
from __future__ import annotations

import csv
import gzip
import re
from collections import defaultdict
from collections.abc import Iterator
from typing import TextIO

GENE_INFO_TAXID_HUMAN = "9606"

# Descriptive names shorter than this, or longer than this many words, are not indexed.
MIN_NAME_LEN = 6
MAX_NAME_WORDS = 8

_WORD_RE = re.compile(r"[a-z0-9]+")

# An upper-case token: hyphenated symbols (NKX2-1, HLA-DRB1) stay whole; a hyphen followed by
# lower case ("ITPR1-related") ends the token so the bare symbol still matches.
UPPERCASE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*(?![A-Za-z0-9])")


def _open_text(path: str) -> TextIO:
    """Open a text file that may be gzip-compressed, detected from its magic bytes."""
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gzipped else open(path, encoding="utf-8")


def normalise(text: str) -> list[str]:
    """Lower-case alphanumeric tokens of ``text``, punctuation dropped."""
    return _WORD_RE.findall(text.lower())


def normalise_phrase(text: str) -> str:
    """``normalise`` joined with single spaces: the key form of every name dictionary."""
    return " ".join(normalise(text))


def split_pipe(value: str | None) -> list[str]:
    """Non-empty parts of a pipe-separated HGNC field, quotes and surrounding space removed."""
    return [x.strip().strip('"') for x in (value or "").strip('"').split("|")
            if x.strip().strip('"')]


def token_pattern(needle: str, ignore_case: bool = False) -> re.Pattern[str]:
    """Regex matching ``needle`` bounded by non-alphanumeric characters (so "P-gp" matches)."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])",
                      re.IGNORECASE if ignore_case else 0)


def _g2p_rows(path: str) -> Iterator[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def g2p_ids(path: str) -> set[str]:
    """Set of ``g2p id`` values in a G2P export CSV."""
    return {(row.get("g2p id") or "").strip() for row in _g2p_rows(path)} - {""}


def g2p_hgnc_ids(path: str) -> dict[str, str]:
    """``g2p id`` to the ``hgnc id`` column as written (with or without the ``HGNC:`` prefix),
    in file order, for rows with a non-empty ``g2p id``."""
    out: dict[str, str] = {}
    for row in _g2p_rows(path):
        gid = (row.get("g2p id") or "").strip()
        if gid:
            out[gid] = (row.get("hgnc id") or "").strip()
    return out


def load_gene_info(path: str) -> dict[str, str]:
    """NCBI GeneID (str) to Symbol for human rows (tax_id 9606) of ``gene_info``."""
    mp: dict[str, str] = {}
    with _open_text(path) as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        i_tax, i_gid, i_sym = (header.index(c) for c in ("tax_id", "GeneID", "Symbol"))
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_tax, i_gid, i_sym) or parts[i_tax] != GENE_INFO_TAXID_HUMAN:
                continue
            mp[parts[i_gid]] = parts[i_sym]
    return mp


def _pubtator_rows(path: str, pmids: set[str] | None) -> Iterator[tuple[str, list[str], set[str], str]]:
    """(pmid, GeneIDs, mention strings, resource) per gene2pubtator3 row, restricted to ``pmids``
    when given. A ``;``-separated GeneID cell yields every id."""
    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pmid = parts[0]
            if pmids is not None and pmid not in pmids:
                continue
            ids = [e.strip() for e in (parts[2] or "").split(";") if e.strip()]
            mentions = {m for m in (parts[3] if len(parts) > 3 else "").split("|") if m}
            yield pmid, ids, mentions, parts[4] if len(parts) > 4 else ""


def load_pubtator_genes(path: str, pmids: set[str] | None, gene_info: dict[str, str]) -> dict[str, set[str]]:
    """pmid to the human gene symbols linked to it by any resource in the gene2pubtator3 file.

    ``pmids=None`` reads the whole file. GeneIDs absent from ``gene_info`` (non-human) are
    skipped. Rows from database cross-references (gene2pubmed, BioGRID, ...) count as well as
    PubTator3 text annotations.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for pmid, ids, _mentions, _resource in _pubtator_rows(path, pmids):
        for eid in ids:
            sym = gene_info.get(eid)
            if sym:
                out[pmid].add(sym)
    return dict(out)


def load_pubtator_gene_ids(path: str, pmids: set[str] | None) -> dict[str, dict[str, set[str]]]:
    """pmid to {NCBI GeneID: mention strings} from rows whose resource includes PubTator3.

    Database cross-reference rows are ignored, so every GeneID returned was annotated in text.
    The caller resolves GeneIDs to genes by identifier.
    """
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for pmid, ids, mentions, resource in _pubtator_rows(path, pmids):
        if "PubTator3" not in resource:
            continue
        for eid in ids:
            out[pmid][eid] |= mentions
    return {p: dict(d) for p, d in out.items()}


def mention_in_text(text: str, mentions: set[str], symbol: str) -> bool:
    """True when any of ``mentions`` (or ``symbol`` when there are none) occurs in ``text`` as a
    whole token, case-insensitive."""
    if not text:
        return False
    return any(token_pattern(needle, ignore_case=True).search(text) for needle in (mentions or {symbol}))


# ------------------------------------------------------------------ symbol-matching primitives

# Symbols never matched verbatim: English words, clinical abbreviations, method acronyms, the
# codon TAT, and symbols of genes on non-DD G2P panels.
SYMBOL_FALLBACK_BLOCKLIST = frozenset({
    "CAT", "WAS", "ACHE", "MARS", "ARC", "BAD", "BID", "CAP", "COIL", "DIP", "FAT", "FLOT",
    "HIP", "IMPACT", "LAMP", "LARGE", "MICE", "OCT", "RAN", "SHE", "SPAG", "TANK", "WARS",
    "APEX", "CRISP", "MASS", "MICAL", "PALM", "SCAN", "SLIT", "TRIP", "WISP", "ARMS", "ATP",
    "DNA", "RNA", "EEG", "MRI", "CNS", "IQ", "ASD", "ADHD", "PCR", "CGH", "SNP", "CNV",
    "NGS", "WES", "WGS", "HPO", "MIM", "OMIM",
    "AIP", "MAX", "MET", "TUB",
    "TAT",
})

# Document-level rule: when one of these lower-case strings occurs anywhere in the abstract, no
# occurrence of the symbol is the gene.
AMBIGUOUS_SYMBOL_CONTEXT: dict[str, tuple[str, ...]] = {
    "CAD": ("coronary artery disease", "coronary atherosclerotic disease", "atheroscler",
            "premature cad", "computer-aided design", "computer aided design",
            "computer-aided diagnosis", "computer aided diagnosis",
            "computer-aided detection", "computer aided detection"),
    "SET": ("short exercise test",),
    "STAR": ("short telomere associated retinopathy", "short telomere-associated retinopathy",
             "star syndrome", "telecanthus"),
}

# Occurrence-level rules: an occurrence followed (or preceded) by the pattern is part of a
# longer protein or product name. The symbol is admitted when any occurrence is free of both.
AMBIGUOUS_SYMBOL_FOLLOWED_BY: dict[str, re.Pattern[str]] = {
    "SET": re.compile(r"\s*(?:binding factor|domain|and MYND|\(BP1\))", re.I),
}
AMBIGUOUS_SYMBOL_PRECEDED_BY: dict[str, re.Pattern[str]] = {
    "KIT": re.compile(r"(?:sequencing|MLPA|PCR|SALSA|extraction|amplification|assay|detection|"
                      r"isolation|purification|labell?ing|hybridi[sz]ation)\s+$", re.I),
}


def uppercase_tokens(text: str) -> set[str]:
    """Distinct upper-case tokens of ``text`` (see ``UPPERCASE_TOKEN_RE``)."""
    return set(UPPERCASE_TOKEN_RE.findall(text))


def symbol_blocked_by_context(symbol: str, text: str, lowered: str) -> bool:
    """True when the context rules show that ``symbol`` in ``text`` is not the gene.

    ``lowered`` is ``text.lower()``, passed in so callers lower-case once per abstract.
    """
    if any(x in lowered for x in AMBIGUOUS_SYMBOL_CONTEXT.get(symbol, ())):
        return True
    after_pat = AMBIGUOUS_SYMBOL_FOLLOWED_BY.get(symbol)
    before_pat = AMBIGUOUS_SYMBOL_PRECEDED_BY.get(symbol)
    if after_pat is None and before_pat is None:
        return False
    for m in token_pattern(symbol).finditer(text):
        if after_pat is not None and after_pat.match(text, m.end()):
            continue
        if before_pat is not None and before_pat.search(text[max(0, m.start() - 40):m.start()]):
            continue
        return False        # this occurrence stands alone, so the symbol is admitted
    return True


def symbol_admissible(symbol: str, text: str, lowered: str) -> bool:
    """A symbol of three or more characters, not blocklisted and not declined by context."""
    return (len(symbol) >= 3 and symbol not in SYMBOL_FALLBACK_BLOCKLIST
            and not symbol_blocked_by_context(symbol, text, lowered))


def symbol_written(symbol: str, text: str, lowered: str) -> bool:
    """True when ``symbol`` occurs in ``text`` case-sensitively as a whole token and is admissible."""
    return (len(symbol) >= 3 and symbol not in SYMBOL_FALLBACK_BLOCKLIST
            and token_pattern(symbol).search(text) is not None
            and not symbol_blocked_by_context(symbol, text, lowered))


# ------------------------------------------------------------------ descriptive-name matching

class GeneNameMatcher:
    """Finds HGNC descriptive gene names in free text and returns the matching symbols.

    Matching is by n-gram lookup over normalised tokens (a few thousand dictionary probes per
    abstract, independent of dictionary size). Only full names are indexed, so a name that is a
    disease label plus an index ("Bardet-Biedl syndrome 1") matches only when written in full.
    """

    def __init__(self, name_to_symbols: dict[str, set[str]]):
        self.name_to_symbols = name_to_symbols
        self.max_words = max((len(k.split()) for k in name_to_symbols), default=1)
        self.max_words = min(self.max_words, MAX_NAME_WORDS)

    @staticmethod
    def _name_variants(name: str) -> list[str]:
        """The forms of an HGNC name that papers write.

        Parenthetical qualifiers and the words homolog/ortholog are removed ("hairless homolog
        (mouse)" gives "hairless"). A slash-joined name of a bifunctional enzyme also contributes
        each slash part of at least two words, with a leading bifunctional/putative/probable/novel
        removed; single-word parts such as "serine" are not indexed.
        """
        base = re.sub(r"\s*\([^)]*\)", " ", name)
        base = re.sub(r"\b(?:homolog|homologue|ortholog|orthologue)\b", " ", base)
        whole = " ".join(base.split())
        out = [whole] if whole else []

        parts = base.split("/")
        if len(parts) < 2:
            return out
        for part in parts:
            part = re.sub(r"^\s*(?:bifunctional|putative|probable|novel)\s+", "", part.strip(),
                          flags=re.I)
            part = " ".join(part.split())
            if part and len(part.split()) >= 2:
                out.append(part)
        return out

    @classmethod
    def from_hgnc(cls, hgnc_path: str, keep_symbols: set[str]) -> GeneNameMatcher:
        """Index the ``name``, ``alias_name`` and ``prev_name`` fields of ``hgnc_complete_set.txt``
        for the genes whose approved symbol is in ``keep_symbols``."""
        name_to_symbols: dict[str, set[str]] = defaultdict(set)
        with _open_text(hgnc_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                symbol = (row.get("symbol") or "").strip()
                if not symbol or symbol not in keep_symbols:
                    continue
                raw: list[str] = []
                for field in ("name", "alias_name", "prev_name"):
                    raw.extend(split_pipe(row.get(field)))
                variants: list[str] = []
                for name in raw:
                    variants.extend(cls._name_variants(name))
                for name in variants:
                    key = normalise_phrase(name)
                    if len(key) < MIN_NAME_LEN or len(key.split()) > MAX_NAME_WORDS:
                        continue
                    name_to_symbols[key].add(symbol)
        return cls(dict(name_to_symbols))

    def find(self, text: str) -> set[str]:
        """Symbols whose indexed name occurs in ``text``. Longer matches take the tokens first,
        so "arginase 1" resolves to ARG1 alone even when a shorter indexed name overlaps it."""
        tokens = normalise(text)
        n = len(tokens)
        hits: set[str] = set()
        covered = [False] * n
        for size in range(min(self.max_words, n), 0, -1):
            for i in range(n - size + 1):
                if any(covered[i:i + size]):
                    continue
                key = " ".join(tokens[i:i + size])
                if len(key) < MIN_NAME_LEN:
                    continue
                found = self.name_to_symbols.get(key)
                if found:
                    hits |= found
                    for j in range(i, i + size):
                        covered[j] = True
        return hits
