"""Gene mention detection for a TIAB, and the G2P candidate set that follows from it.

Two complementary sources, because they fail differently:

1. **PubTator3 symbol NER** (``gene2pubtator3``), filtered to human via ``gene_info``.
   Best-in-class for gene *symbols* (~86% F1 end-to-end) and available as a full-corpus bulk
   download, so no API and no local NER run. This is the primary source.

2. **HGNC descriptive-name matching** over the TIAB text. This exists to catch what (1) misses:
   papers that discuss the gene product by **protein or enzyme name and never write the
   symbol**. From our own miss set, PMID 2913054 -- "Differential expression of the two human
   arginase genes in hyperargininemia" -- is a genuine ARG1 paper whose abstract says
   "arginase", "AI", "AII" and never "ARG1". That register (older, more mechanistic) is exactly
   where the screen is already weakest, so the two failure modes compound unless corrected.

   **Names only, never symbols.** Symbol matching is where gene-name ambiguity lives -- CAT,
   SET, MAX, WAS, T, ACHE, STAR, REST, MARS, HR, AIP collide with English and clinical
   vocabulary; 0.57% of official symbols are English words and including alias symbols raises
   intra-species ambiguity from 0.02% to 5.02%. PubTator already handles symbols well, so a
   symbol dictionary would add ambiguity without adding recall. Descriptive names ("arginase 1",
   "fibrillin 1") are long and specific, so this stays high-precision.

   The dictionary is restricted to the **genes present in the G2P panel** (~2,552), not all of
   HGNC (~43,000), which cuts both the size and the false-match surface substantially.

**Matching is on the FULL name by default.** An earlier version also indexed a "family stem"
formed by stripping a trailing index from the HGNC name, so that "arginase" would match
"arginase 1"/"arginase 2". That is wrong in general, because a large share of HGNC gene names
embed the *disease* rather than the protein: "Bardet-Biedl syndrome 1", "Bardet-Biedl syndrome
2", ... Stripping the index turns the stem into a disease name, so any paper mentioning
Bardet-Biedl syndrome matched all eleven BBS genes. That is disease matching wearing gene
matching's clothes, and it inflates the candidate set with entries the abstract never named.
Requiring the full name ("Bardet-Biedl syndrome 1" for BBS1) removes the failure.

``family_stems=True`` restores the old behaviour for measurement, and ``FAMILY_STEM_BLOCKLIST``
holds the disease-ish head nouns that must never form a stem even then.

Provenance is carried per candidate (``symbol_match`` / ``name_match`` / ``fallback``) so the
precision audit can report each source separately, and a low-precision source can be
down-weighted later without re-running the stage.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict

GENE_INFO_TAXID_HUMAN = "9606"


def _open_text(path: str):
    """Open a possibly-gzipped text file. The PubTator bulk dumps ship both ways."""
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gzipped else open(path, encoding="utf-8")

# Names shorter than this are not used, even from HGNC: single short words are the ambiguous
# case the name dictionary is specifically avoiding.
MIN_NAME_LEN = 6
MAX_NAME_WORDS = 8

_WORD_RE = re.compile(r"[a-z0-9]+")
# trailing family index: "arginase 1", "filamin B"
_FAMILY_SUFFIX_RE = re.compile(r"\s+(?:[0-9]+|[ivx]+|[a-z])$")

# A stem ending in one of these is a disease/phenotype label, not a protein name, so it must
# never stand in for the gene family -- "Bardet-Biedl syndrome" is not a gene.
FAMILY_STEM_BLOCKLIST = (
    "syndrome", "disease", "disorder", "deficiency", "dysplasia", "dystrophy", "anomaly",
    "atrophy", "malformation", "susceptibility", "type", "complex", "epilepsy", "ataxia",
    "retardation", "hypoplasia", "aplasia", "neuropathy", "myopathy", "carcinoma", "cancer",
    "tumor", "tumour", "encephalopathy", "degeneration", "sclerosis", "palsy", "seizures",
)


def normalise(text: str) -> list[str]:
    """Lowercase word tokens, punctuation stripped."""
    return _WORD_RE.findall(text.lower())


def load_gene_info(path: str) -> dict[str, str]:
    """NCBI GeneID (str) -> canonical Symbol, restricted to human (tax_id 9606)."""
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


def load_pubtator_genes(
    path: str, pmids: set[str] | None, gene_info: dict[str, str]
) -> dict[str, set[str]]:
    """pmid -> set of human gene symbols annotated by PubTator3.

    ``pmids=None`` reads the whole file. Unlike the earlier implementation this keeps **every**
    GeneID in a multi-id cell rather than only the first -- a ``12345;67890`` annotation names
    two genes and dropping the second silently loses candidates.
    """
    out: dict[str, set[str]] = defaultdict(set)
    with _open_text(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pmid = parts[0]
            if pmids is not None and pmid not in pmids:
                continue
            for eid in (parts[2] or "").split(";"):
                sym = gene_info.get(eid.strip())
                if sym:
                    out[pmid].add(sym)
    return dict(out)


class GeneNameMatcher:
    """Matches HGNC descriptive gene names in free text, restricted to a symbol whitelist.

    Uses n-gram lookup rather than a regex alternation: for a ~300-word abstract this is a few
    thousand dict probes, independent of dictionary size, and needs no extra dependency.
    """

    def __init__(self, name_to_symbols: dict[str, set[str]],
                 family_to_symbols: dict[str, set[str]] | None = None):
        family_to_symbols = family_to_symbols or {}
        self.name_to_symbols = name_to_symbols
        self.family_to_symbols = family_to_symbols
        self.max_words = max((len(k.split()) for k in name_to_symbols), default=1)
        self.max_words = min(self.max_words, MAX_NAME_WORDS)

    @classmethod
    def from_hgnc(cls, hgnc_path: str, keep_symbols: set[str],
                  family_stems: bool = False) -> "GeneNameMatcher":
        """Build from `hgnc_complete_set.txt`, keeping only genes in `keep_symbols`.

        Download: https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
        """
        import csv as _csv

        name_to_symbols: dict[str, set[str]] = defaultdict(set)
        family_to_symbols: dict[str, set[str]] = defaultdict(set)
        with _open_text(hgnc_path) as f:
            for row in _csv.DictReader(f, delimiter="\t"):
                symbol = (row.get("symbol") or "").strip()
                if not symbol or symbol not in keep_symbols:
                    continue
                raw: list[str] = []
                for field in ("name", "alias_name", "prev_name"):
                    val = (row.get(field) or "").strip().strip('"')
                    raw.extend(v.strip().strip('"') for v in val.split("|") if v.strip())
                for name in raw:
                    key = " ".join(normalise(name))
                    if len(key) < MIN_NAME_LEN or len(key.split()) > MAX_NAME_WORDS:
                        continue
                    name_to_symbols[key].add(symbol)
                    if not family_stems:
                        continue
                    fam = _FAMILY_SUFFIX_RE.sub("", key)
                    if (fam != key and len(fam) >= MIN_NAME_LEN
                            and not fam.endswith(FAMILY_STEM_BLOCKLIST)):
                        family_to_symbols[fam].add(symbol)
        return cls(dict(name_to_symbols), dict(family_to_symbols))

    def find(self, text: str) -> set[str]:
        """Gene symbols whose descriptive name (or family stem) appears in `text`.

        Longest match wins: "arginase 1" resolves to ARG1 alone, and the family stem
        "arginase" is only consulted where no longer exact name covered those tokens. Without
        this, an abstract naming the specific gene would still drag in every sibling.
        """
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
                found = self.name_to_symbols.get(key) or self.family_to_symbols.get(key)
                if found:
                    hits |= found
                    for j in range(i, i + size):
                        covered[j] = True
        return hits
