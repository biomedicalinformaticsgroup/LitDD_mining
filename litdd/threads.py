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

import gzip

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
    # The 2025-02-15 export (the panel the annotated set was labelled against) carried this
    # field as "molecular mechanism categorisation" (values inferred/evidence); the 2026
    # exports renamed it "molecular mechanism support" and re-used "categorisation" for a
    # different column. Preference order keeps the 2026 exports on the right column; the
    # fallback lets the annotated-set threads be re-rendered byte-identically.
    ("molecular_mechanism_support", ("molecular mechanism support",
                                     "molecular_mechanism_support",
                                     "molecular mechanism categorisation")),
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


def load_gene_names(gene_info_path: str) -> dict[str, str]:
    """``{key: full gene name}`` from an NCBI ``gene_info`` dump (gzipped TSV).

    Keys are both the official symbol (``ARG1``) and the numeric HGNC id (``"603"``),
    so a G2P row can be resolved by ``hgnc id`` first and ``gene symbol`` as fallback.
    The name is ``description`` (e.g. ``arginase 1``) — the field a TIAB that says
    "arginase" can align against when the thread's symbol alone cannot.
    """
    names: dict[str, str] = {}
    with gzip.open(gene_info_path, "rt") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            row = line.rstrip("\n").split("\t")
            desc = row[idx["description"]]
            if not desc or desc == "-":
                continue
            symbol = row[idx["Symbol"]]
            if symbol and symbol != "-":
                names.setdefault(symbol, desc)
            for xref in row[idx["dbXrefs"]].split("|"):
                # gene_info renders the HGNC xref as "HGNC:HGNC:603"
                if xref.startswith("HGNC:"):
                    names.setdefault(xref.split(":")[-1], desc)
    return names


def _gene_name_for_row(row, resolved: dict[str, str | None], gene_names: dict[str, str]) -> str:
    """Full gene name for one G2P row: HGNC id first, symbol as fallback, '' if unknown."""
    hgnc_col = resolved.get("hgnc_id")
    if hgnc_col is not None and pd.notna(row[hgnc_col]):
        v = row[hgnc_col]
        key = str(int(v)) if isinstance(v, float) and v == int(v) else str(v).replace("HGNC:", "")
        if key in gene_names:
            return gene_names[key]
    sym_col = resolved.get("gene_symbol")
    if sym_col is not None and pd.notna(row[sym_col]):
        return gene_names.get(str(row[sym_col]), "")
    return ""


def build_lgmde_map(g2p_csv: str, strict: bool = True,
                    gene_names: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``{g2p_id: lgmde_thread}`` for every entry in the export.

    ``gene_names`` (from :func:`load_gene_names`) selects the *gene-name* thread
    variant: the full gene name is inserted as an extra field directly after
    ``previous gene symbols``, so gene-identifying text stays contiguous. With the
    default ``None`` the rendering is byte-identical to what the released
    cross-encoder was fine-tuned on — do not pass ``gene_names`` on the deployment
    path unless the deployed model was trained on that variant.
    """
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
    resolved_by_field = dict(resolved)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        values = []
        for field, col in resolved:
            if col is None:
                values.append("")
            else:
                v = row[col]
                # Training rendered the HGNC id with its prefix and no float artefact.
                # Older exports (2025-02-15) already carry the prefix as a string; do not
                # double it, or the annotated-set threads stop matching the panel.
                if field == "hgnc_id" and pd.notna(v) and not str(v).startswith("HGNC:"):
                    v = f"HGNC:{int(v) if isinstance(v, float) and v == int(v) else v}"
                values.append(str(v))
            if field == "previous_gene_symbols" and gene_names is not None:
                values.append(_gene_name_for_row(row, resolved_by_field, gene_names))
        out[str(row[id_col])] = SEPARATOR.join(values)
    return out


def build_lgmde_list(g2p_csv: str, strict: bool = True,
                     gene_names: dict[str, str] | None = None) -> list[str]:
    """Unique LGMDE threads, for use as the cross-encoder candidate pool."""
    return sorted(set(build_lgmde_map(g2p_csv, strict=strict, gene_names=gene_names).values()))
