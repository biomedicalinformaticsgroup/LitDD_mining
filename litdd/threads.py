"""Single source of truth for the G2P "LGMDE thread" string.

Why this module exists
----------------------
The thread is the text the cross-encoder scores against a TIAB, so training and inference
must render it *identically* -- the model is fine-tuned on the exact string format. They did
not: training built it positionally from the first 15 CSV columns with pandas defaults, while
inference used a hard-coded list of column *names* read with ``keep_default_na=False``. Three
concrete divergences resulted, for the same G2P entry:

    training  : G2P00410 - NF1 - 613113.0 - HGNC:7765 - nan - ... - absent gene product - nan
                - loss of function - inferred
    inference : G2P00410 - NF1 - 613113   - 7765      -     - ... -                     -
                - loss of function -

  1. ``"inferred variant consequence"`` is not a column in any G2P export (the column is
     ``"variant consequence"``), so at inference that field was **always blank** while
     training had it populated.
  2. Missing values rendered as ``"nan"`` in training and ``""`` at inference; ``gene mim``
     as ``613113.0`` vs ``613113``; ``hgnc id`` as ``HGNC:7765`` vs ``7765``.
  3. The 15th field was ``molecular mechanism support`` in training but
     ``molecular mechanism categorisation`` at inference.

The released cross-encoder was fine-tuned on the training rendering, so that is the format
this module reproduces and both sides now call.

Column *positions* are not stable across G2P releases either (the 2026 export renamed and
reordered fields), so `build_lgmde_map` resolves by name with explicit aliases and fails loudly
on an unresolvable field rather than silently emitting a blank.
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
    # NOTE: "inferred variant consequence" is the name the old inference path looked for and
    # is not present in any export -- this is the field that was silently blank.
    ("variant_consequence", ("variant consequence", "inferred variant consequence",
                             "variant_consequence")),
    ("variant_types", ("variant types", "variant_types")),
    ("molecular_mechanism", ("molecular mechanism", "molecular_mechanism")),
    ("molecular_mechanism_support", ("molecular mechanism support",
                                     "molecular_mechanism_support")),
]

SEPARATOR = " - "


def _resolve(columns, accepted: tuple[str, ...]) -> str | None:
    lookup = {c.strip().lower(): c for c in columns}
    for name in accepted:
        hit = lookup.get(name.strip().lower())
        if hit is not None:
            return hit
    return None


def load_g2p(g2p_csv: str) -> pd.DataFrame:
    """Read a G2P export preserving the training-time rendering of missing values.

    Deliberately uses pandas defaults (NaN, float coercion) because the released
    cross-encoder was fine-tuned on strings produced that way -- ``"nan"``, ``613113.0``.
    """
    df = pd.read_csv(g2p_csv)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_lgmde_map(g2p_csv: str, strict: bool = True) -> dict[str, str]:
    """Return ``{g2p_id: lgmde_thread}`` for every entry in the export."""
    df = load_g2p(g2p_csv)
    resolved: list[tuple[str, str | None]] = [
        (field, _resolve(df.columns, accepted)) for field, accepted in LGMDE_FIELDS
    ]
    missing = [f for f, col in resolved if col is None]
    if missing and strict:
        raise KeyError(
            f"G2P export {g2p_csv} is missing column(s) for LGMDE field(s) {missing}. "
            "Refusing to emit threads with silently blank fields -- the cross-encoder is "
            "fine-tuned on the full string. Add an alias to LGMDE_FIELDS if the export "
            "renamed the column."
        )

    id_col = _resolve(df.columns, ("g2p id", "g2p_id"))
    if id_col is None:
        raise KeyError(f"{g2p_csv} has no g2p id column")
    df = df.drop_duplicates(id_col)

    # str() on the raw cell reproduces the training rendering exactly: NaN -> "nan",
    # numeric MIMs -> "613113.0". Do not "clean" these without re-finetuning the
    # cross-encoder, which was fine-tuned on strings produced this way.
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        values = []
        for field, col in resolved:
            if col is None:
                values.append("")
                continue
            v = row[col]
            # Training rendered the HGNC id with its prefix and no float artefact.
            if field == "hgnc_id" and pd.notna(v):
                v = f"HGNC:{int(v) if isinstance(v, float) and v == int(v) else v}"
            values.append(str(v))
        out[str(row[id_col])] = SEPARATOR.join(values)
    return out


def build_lgmde_list(g2p_csv: str, strict: bool = True) -> list[str]:
    """Unique LGMDE threads, for use as the cross-encoder candidate pool."""
    return sorted(set(build_lgmde_map(g2p_csv, strict=strict).values()))
