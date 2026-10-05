"""Render the G2P entry label used in the annotated datasets and evaluation fixtures.

The label is the fifteen entry fields of a G2P export joined by `` - ``, in a fixed order,
with pandas' default rendering of missing values (``nan``) and of numeric columns that
contain blanks (``613113.0``). Columns are resolved by name with aliases for the export
layouts in use, and a missing column raises rather than rendering an empty field.
"""
from __future__ import annotations

import pandas as pd

# (canonical field, accepted column names in order of preference)
LGMDE_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("g2p_id", ("g2p id", "g2p_id")),
    ("gene_symbol", ("gene symbol", "gene_symbol")),
    ("gene_mim", ("gene mim", "gene_mim")),
    ("hgnc_id", ("hgnc id", "hgnc_id")),
    ("previous_gene_symbols", ("previous gene symbols", "previous_gene_symbols")),
    ("disease_name", ("disease name", "disease_name")),
    ("disease_mim", ("disease mim", "disease_mim")),
    ("disease_mondo", ("disease MONDO", "disease mondo", "disease_mondo")),
    ("allelic_requirement", ("allelic requirement", "allelic_requirement")),
    ("cross_cutting_modifier", ("cross cutting modifier", "cross_cutting_modifier")),
    ("confidence", ("confidence",)),
    ("variant_consequence", ("variant consequence", "inferred variant consequence",
                             "variant_consequence")),
    ("variant_types", ("variant types", "variant_types")),
    ("molecular_mechanism", ("molecular mechanism", "molecular_mechanism")),
    # Exports before 2026 name this column "molecular mechanism categorisation"; later
    # exports use "molecular mechanism support" and reuse "categorisation" for another column.
    ("molecular_mechanism_support", ("molecular mechanism support",
                                     "molecular_mechanism_support",
                                     "molecular mechanism categorisation")),
]

SEPARATOR = " - "


def _resolve(columns, accepted: tuple[str, ...]) -> str | None:
    """The first column name in ``accepted`` present in ``columns``, compared case-insensitively."""
    lookup = {c.strip().lower(): c for c in columns}
    for name in accepted:
        hit = lookup.get(name.strip().lower())
        if hit is not None:
            return hit
    return None


def load_g2p(g2p_csv: str) -> pd.DataFrame:
    """Read a G2P export with pandas defaults (NaN for blanks, float columns where a numeric
    column contains blanks) and stripped column names."""
    df = pd.read_csv(g2p_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_lgmde_map(g2p_csv: str, strict: bool = True) -> dict[str, str]:
    """``{g2p_id: label}`` for every entry in the export.

    With ``strict`` a field whose column is absent raises ``KeyError``; otherwise the field
    renders as an empty string. The HGNC id is rendered with its ``HGNC:`` prefix."""
    df = load_g2p(g2p_csv)
    resolved: list[tuple[str, str | None]] = [
        (field, _resolve(df.columns, accepted)) for field, accepted in LGMDE_FIELDS
    ]
    missing = [f for f, col in resolved if col is None]
    if missing and strict:
        raise KeyError(
            f"G2P export {g2p_csv} is missing column(s) for LGMDE field(s) {missing}; "
            "add an alias to LGMDE_FIELDS if the export renamed the column"
        )

    id_col = _resolve(df.columns, ("g2p id", "g2p_id"))
    if id_col is None:
        raise KeyError(f"{g2p_csv} has no g2p id column")
    df = df.drop_duplicates(id_col)

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        values = []
        for field, col in resolved:
            if col is None:
                values.append("")
                continue
            v = row[col]
            if field == "hgnc_id" and pd.notna(v) and not str(v).startswith("HGNC:"):
                v = f"HGNC:{int(v) if isinstance(v, float) and v == int(v) else v}"
            values.append(str(v))
        out[str(row[id_col])] = SEPARATOR.join(values)
    return out


def build_lgmde_list(g2p_csv: str, strict: bool = True) -> list[str]:
    """Sorted unique labels of the export."""
    return sorted(set(build_lgmde_map(g2p_csv, strict=strict).values()))
