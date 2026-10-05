"""Build the contextualised G2P threads read by the adjudication stage.

Reads a Gene2Phenotype all-panel CSV export, the MONDO and HPO ontologies as OBO files and
the packaged disease-name alias table (``pipeline/data/disease_name_aliases.tsv``). Writes a
JSON object mapping every G2P identifier in the export to a multi-line text block, plus a
``__provenance__`` entry naming the inputs. ``litdd.pipeline.llm_map --threads context
--context_json`` reads this file to render the candidate entries in the adjudication prompt,
so the export it is built from has to be the one the candidates were resolved against.

Each block has the layout (thread format ``v1``)::

    G2P ID: G2P00001
    Gene Symbol: HMX1
    Disease Name: HMX1-related oculoauricular syndrome
    Previous Gene Symbols (former names of this gene, not disease names): H6; NKX5-3
    Disease Synonyms: OCACS; oculoauricular syndrome
    Disease Definition: <MONDO definition>
    Phenotypes: Autosomal recessive inheritance; Microcornea; ...
    Allelic Requirement: biallelic_autosomal
    Molecular Mechanism: loss of function
    Variant Consequence: absent gene product

The disease synonyms and definition come from the MONDO term named in the export's
``disease MONDO`` column; the MONDO label joins the synonyms and the export's own disease name
is excluded from them. The phenotypes are the HPO term names of the export's ``phenotypes``
column, in the export's order. A field without a value renders as ``None``. The previous-symbol
line lists the export's previous gene symbols minus any that the alias table records as a
disease name for that gene, and is omitted when no symbol remains.

Ported from the contextualised-thread builder written by Fabian Rott.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pronto

logger = logging.getLogger(__name__)

THREAD_FORMAT = "v1"
PREVIOUS_LABEL_IN = "Previous Gene Symbols:"
PREVIOUS_LABEL_OUT = "Previous Gene Symbols (former names of this gene, not disease names):"
NULL_SYMBOLS = ("none", "nan")

_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_ALIASES_TSV = _DATA_DIR / "disease_name_aliases.tsv"

# Export columns (after normalisation) read by the builder.
G2P_COLUMNS = (
    "g2p_id",
    "gene_symbol",
    "disease_name",
    "previous_gene_symbols",
    "allelic_requirement",
    "molecular_mechanism",
    "variant_consequence",
    "disease_mondo",
    "phenotypes",
)


@dataclass
class MondoTerm:
    """The fields of one MONDO term used in a thread."""

    name: str | None
    definition: str | None
    synonyms: list[str] = field(default_factory=list)


@dataclass
class ContextThread:
    """One rendered candidate entry; ``None`` fields render as the string ``None``."""

    g2p_id: object
    gene_symbol: object
    disease_name: object
    previous_gene_symbols: list[str]
    disease_synonyms: list[str]
    disease_definition: str | None
    phenotypes: list[str]
    allelic_requirement: str | None
    molecular_mechanism: str | None
    variant_consequence: str | None

    def render(self) -> str:
        """Return the thread in format ``v1`` with a trailing newline."""
        join = lambda items: "; ".join(items) if items else None  # noqa: E731
        return (
            f"G2P ID: {self.g2p_id}\n"
            f"Gene Symbol: {self.gene_symbol}\n"
            f"Disease Name: {self.disease_name}\n"
            f"{PREVIOUS_LABEL_IN} {join(self.previous_gene_symbols)}\n"
            f"Disease Synonyms: {join(self.disease_synonyms)}\n"
            f"Disease Definition: {self.disease_definition}\n"
            f"Phenotypes: {join(self.phenotypes)}\n"
            f"Allelic Requirement: {self.allelic_requirement}\n"
            f"Molecular Mechanism: {self.molecular_mechanism}\n"
            f"Variant Consequence: {self.variant_consequence}\n"
        )


def read_g2p_export(path: str | Path) -> pd.DataFrame:
    """Read a G2P CSV export with column names lower-cased and spaces replaced by underscores.

    Exports that name the consequence column ``inferred variant consequence`` are mapped onto
    ``variant_consequence``. Columns the builder reads but the export lacks are added empty.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.replace(" ", "_").lower() for c in df.columns})
    if "variant_consequence" not in df.columns and "inferred_variant_consequence" in df.columns:
        df["variant_consequence"] = df["inferred_variant_consequence"]
    for col in G2P_COLUMNS:
        if col not in df.columns:
            logger.warning("export lacks column %r; the field renders as None", col)
            df[col] = None
    return df


def _split_ids(value: object) -> list[str]:
    """Split a ``;``-separated cell into stripped, non-empty tokens; blanks give an empty list."""
    if not isinstance(value, str):
        return []
    return [token.strip() for token in value.split(";") if token.strip()]


def _text_or_none(value: object) -> str | None:
    """Return a non-empty string cell unchanged and anything else (NaN, blank) as ``None``."""
    return value if isinstance(value, str) and value else None


def obo_release(path: str | Path) -> str | None:
    """Return the ``data-version`` header of an OBO file, or ``None`` when the header is absent."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("["):
                break
    return None


def lookup_mondo(ontology: pronto.Ontology, mondo_ids: list[str]) -> tuple[dict[str, MondoTerm], list[str]]:
    """Resolve MONDO identifiers to their label, definition and synonyms.

    Returns the resolved terms keyed by identifier and the list of identifiers the ontology
    does not contain.
    """
    terms: dict[str, MondoTerm] = {}
    unresolved: list[str] = []
    for mondo_id in mondo_ids:
        try:
            term = ontology[mondo_id]
        except KeyError:
            logger.warning("MONDO id not found: %s", mondo_id)
            unresolved.append(mondo_id)
            continue
        terms[mondo_id] = MondoTerm(
            name=term.name,
            definition=str(term.definition) if term.definition else None,
            synonyms=sorted(syn.description for syn in term.synonyms),
        )
    logger.info("resolved %d MONDO ids, %d unresolved", len(terms), len(unresolved))
    return terms, unresolved


def lookup_hpo(ontology: pronto.Ontology, hpo_ids: list[str]) -> tuple[dict[str, str], list[str]]:
    """Resolve HPO identifiers to term names; returns the names by identifier and the unresolved ids."""
    names: dict[str, str] = {}
    unresolved: list[str] = []
    for hpo_id in hpo_ids:
        try:
            names[hpo_id] = ontology[hpo_id].name
        except KeyError:
            logger.warning("HPO id not found: %s", hpo_id)
            unresolved.append(hpo_id)
    logger.info("resolved %d HPO ids, %d unresolved", len(names), len(unresolved))
    return names, unresolved


def enrich_export(df: pd.DataFrame, mondo_terms: dict[str, MondoTerm], hpo_names: dict[str, str]) -> pd.DataFrame:
    """Attach the MONDO fields and HPO term names to each export row.

    Adds ``mondo_name``, ``mondo_definition``, ``mondo_synonyms`` and ``hpo_term_names``; rows
    whose identifiers did not resolve get ``None`` (unresolved HPO ids are skipped).
    """
    out = df.copy()
    resolved = [mondo_terms.get(mid) if isinstance(mid, str) else None for mid in out["disease_mondo"]]
    out["mondo_name"] = [t.name if t else None for t in resolved]
    out["mondo_definition"] = [t.definition if t else None for t in resolved]
    out["mondo_synonyms"] = [t.synonyms if t else None for t in resolved]
    out["hpo_term_names"] = [
        [hpo_names[h] for h in _split_ids(cell) if h in hpo_names] or None for cell in out["phenotypes"]
    ]
    return out


def build_threads(enriched: pd.DataFrame) -> list[ContextThread]:
    """Turn the enriched export into thread objects, one per row."""
    threads: list[ContextThread] = []
    for row in enriched.to_dict(orient="records"):
        disease_name = row["disease_name"]
        candidates = list(row["mondo_synonyms"] or [])
        if isinstance(row["mondo_name"], str):
            candidates.append(row["mondo_name"])
        synonyms = sorted({s for s in candidates if s != disease_name})
        threads.append(
            ContextThread(
                g2p_id=row["g2p_id"],
                gene_symbol=row["gene_symbol"],
                disease_name=disease_name,
                previous_gene_symbols=[s.strip() for s in row["previous_gene_symbols"].split(";")]
                if isinstance(row["previous_gene_symbols"], str)
                else [],
                disease_synonyms=synonyms,
                disease_definition=_text_or_none(row["mondo_definition"]),
                phenotypes=list(row["hpo_term_names"] or []),
                allelic_requirement=_text_or_none(row["allelic_requirement"]),
                molecular_mechanism=_text_or_none(row["molecular_mechanism"]),
                variant_consequence=_text_or_none(row["variant_consequence"]),
            )
        )
    return threads


def load_disease_aliases(path: str | Path) -> dict[str, set[str]]:
    """Read the alias table (columns ``kind``, ``gene``, ``alias``, ...) as upper-cased aliases per gene."""
    aliases: dict[str, set[str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            aliases.setdefault(row["gene"], set()).add(row["alias"].upper())
    return aliases


def relabel_previous_symbols(block: str, gene: str, aliases: dict[str, set[str]]) -> str:
    """Rewrite the previous-symbol line of a rendered block.

    The line is relabelled, the symbols recorded as disease names for ``gene`` are removed
    along with ``None``/``nan`` placeholders and duplicates, and the line is dropped when
    nothing remains. Returns the block without a trailing newline.
    """
    blocked = aliases.get(gene, set())
    lines: list[str] = []
    for line in block.splitlines():
        if not line.startswith(PREVIOUS_LABEL_IN):
            lines.append(line)
            continue
        symbols = [s.strip() for s in line.split(":", 1)[1].split(";") if s.strip()]
        kept = [s for s in symbols if s.lower() not in NULL_SYMBOLS and s.upper() not in blocked]
        if kept:
            lines.append(f"{PREVIOUS_LABEL_OUT} {'; '.join(dict.fromkeys(kept))}")
    return "\n".join(lines)


def build_context_threads(
    g2p_csv: str | Path,
    mondo_obo: str | Path,
    hp_obo: str | Path,
    aliases_tsv: str | Path = DEFAULT_ALIASES_TSV,
) -> tuple[dict[str, str], dict[str, object], pd.DataFrame]:
    """Build the thread mapping for one export.

    Returns ``(threads, provenance, enriched)``: the ``{g2p_id: block}`` mapping, the
    provenance entry and the enriched export frame.
    """
    df = read_g2p_export(g2p_csv)
    logger.info("export %s: %d entries", g2p_csv, len(df))

    mondo_ids = sorted({m for m in df["disease_mondo"] if isinstance(m, str)})
    mondo_terms, unresolved_mondo = lookup_mondo(pronto.Ontology(str(mondo_obo), encoding="utf-8"), mondo_ids)
    hpo_ids = sorted({h for cell in df["phenotypes"] for h in _split_ids(cell)})
    hpo_names, unresolved_hpo = lookup_hpo(pronto.Ontology(str(hp_obo), encoding="utf-8"), hpo_ids)

    enriched = enrich_export(df, mondo_terms, hpo_names)
    aliases = load_disease_aliases(aliases_tsv)
    threads = {
        str(t.g2p_id): relabel_previous_symbols(t.render(), str(t.gene_symbol).strip(), aliases)
        for t in build_threads(enriched)
    }
    provenance: dict[str, object] = {
        "builder": "litdd.pipeline.build_context_threads",
        "panel": Path(g2p_csv).name,
        "mondo": obo_release(mondo_obo),
        "hpo": obo_release(hp_obo),
        "format": THREAD_FORMAT,
        "aliases": Path(aliases_tsv).name,
        "unresolved_mondo_ids": unresolved_mondo,
        "unresolved_hpo_ids": unresolved_hpo,
    }
    return threads, provenance, enriched


def write_context_json(path: str | Path, threads: dict[str, str], provenance: dict[str, object]) -> None:
    """Write the provenance entry followed by the threads, in export order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"__provenance__": provenance, **threads}, fh, indent=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the contextualised G2P threads JSON for the adjudication stage.")
    p.add_argument("--g2p_csv", required=True, help="G2P all-panel CSV export the candidates were resolved against")
    p.add_argument("--mondo_obo", required=True, help="MONDO ontology, OBO format")
    p.add_argument("--hp_obo", required=True, help="Human Phenotype Ontology, OBO format")
    p.add_argument("--aliases_tsv", default=str(DEFAULT_ALIASES_TSV), help="disease-name alias table (TSV)")
    p.add_argument("--out_json", required=True, help="output JSON ({g2p_id: block} plus __provenance__)")
    p.add_argument("--keep_parquet", default=None, help="also write the enriched export frame to this parquet path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    threads, provenance, enriched = build_context_threads(args.g2p_csv, args.mondo_obo, args.hp_obo, args.aliases_tsv)
    write_context_json(args.out_json, threads, provenance)
    logger.info("wrote %s (%d threads)", args.out_json, len(threads))
    if args.keep_parquet:
        Path(args.keep_parquet).parent.mkdir(parents=True, exist_ok=True)
        enriched.to_parquet(args.keep_parquet)
        logger.info("wrote %s", args.keep_parquet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
