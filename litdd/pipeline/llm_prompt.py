"""Prompt construction for the adjudication stage.

Inputs: a shard row's ``tiab`` and ``candidates`` (a list of G2P ids), the contextualised
thread JSON (``{g2p_id: multi-line block}``) and the all-panel G2P export used to add
same-gene entries from other panels and to label each candidate with its panel.

Output: the prompt string sent to the model, built from ``prompts/decision_rubric.txt`` with
the placeholders ``{n}``, ``{plural}``, ``{tiab}`` and ``{candidate_lines}`` filled in.

The order of operations in ``llm_map._prepare_shard`` is: ``candidate_list`` normalises the
cell, ``add_panel_siblings`` appends the other-panel entries of the candidate genes as bare
ids, ``contextualise`` swaps every id for its context block, ``label_panel`` appends the
``G2P Panel`` line and ``build_llm_prompt`` renders the numbered list into the template.
"""
from __future__ import annotations

import functools
import json
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from litdd.pipeline.llm_answer import G2P_ID_RE

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
DEFAULT_PROMPT_FILE = os.path.join(PROMPT_DIR, "decision_rubric.txt")


# --------------------------------------------------------------------------------------
# Template
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=8)
def load_prompt_template(path: str = DEFAULT_PROMPT_FILE) -> str:
    """Read a prompt template and check that it has the two required placeholders.

    Placeholders: ``{n}``, ``{plural}``, ``{tiab}``, ``{candidate_lines}``. Raises
    ``ValueError`` when ``{tiab}`` or ``{candidate_lines}`` is absent.
    """
    with open(path, encoding="utf-8") as f:
        template = f.read()
    if "{tiab}" not in template:
        raise ValueError(f"prompt template {path} lacks the {{tiab}} placeholder")
    if "{candidate_lines}" not in template:
        raise ValueError(f"prompt template {path} lacks the {{candidate_lines}} placeholder")
    return template


def fill_placeholders(template: str, fields: dict[str, Any]) -> str:
    """Substitute the named ``{key}`` placeholders and leave every other brace untouched.

    ``str.format`` would interpret any brace in the template, so the substitution is a
    plain string replacement of each known token.
    """
    out = template
    for key, value in fields.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, str(value))
    return out


def render_candidates(candidate_lines: Iterable[str]) -> str:
    """Number the candidate blocks ``1) ...``, one per line group, in the given order."""
    candidate_lines = list(candidate_lines)
    return "\n".join(f"{i + 1}) {c}" for i, c in enumerate(candidate_lines))


def build_llm_prompt(tiab: str, candidate_lines: Iterable[str],
                     template_path: str = DEFAULT_PROMPT_FILE) -> str:
    """Render the adjudication prompt for one TIAB and its candidate blocks.

    The template states the actual number of candidates (``{n}``, ``{plural}``) and the
    candidates are numbered so that multi-line blocks have unambiguous boundaries.
    Raises ``ValueError`` on an empty candidate list: such a prompt would carry no
    candidates and could only be answered NO MATCH, which is indistinguishable from a
    negative. Callers filter or record those rows instead of sending them to the model.
    """
    candidate_lines = list(candidate_lines)
    n = len(candidate_lines)
    if n == 0:
        raise ValueError(
            "build_llm_prompt called with no candidate threads. This would produce a "
            "prompt containing zero candidates and an unconditional 'NO MATCH' answer. "
            "Filter these rows out upstream, or record them as no-candidate rather than "
            "sending them to the LLM."
        )
    plural = "thread" if n == 1 else "threads"
    numbered = render_candidates(candidate_lines)
    tmpl = load_prompt_template(template_path)
    return fill_placeholders(tmpl, {"n": n, "plural": plural, "tiab": tiab,
                                    "candidate_lines": numbered})


# --------------------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------------------
def candidate_list(x: Any) -> list[str]:
    """Normalise a ``candidates`` cell into a list of non-empty id strings.

    Accepts a list or tuple, a numpy array, a pyarrow scalar or ``None``/NaN. Items that
    are empty after stripping are dropped; order is preserved.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    try:
        import pyarrow as pa
        if isinstance(x, pa.Scalar):
            return candidate_list(x.as_py())
    except Exception:  # noqa: BLE001
        pass
    return []


def candidate_ids(candidate_lines: Iterable[str]) -> list[str | None]:
    """The first G2P id found in each candidate block (``None`` when a block has none)."""
    ids = []
    for c in candidate_lines:
        m = G2P_ID_RE.search(str(c))
        ids.append(m.group(0) if m else None)
    return ids


# --------------------------------------------------------------------------------------
# Contextualised threads
# --------------------------------------------------------------------------------------
def load_context_threads(path: str) -> dict[str, str]:
    """Read ``{g2p_id: contextualised multi-line thread}`` from the context JSON.

    Keys starting with ``__`` (provenance) are skipped. Lines whose value is the literal
    ``None`` (an empty enrichment field) are dropped.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        lines = [ln for ln in str(v).splitlines() if not ln.rstrip().endswith(": None")]
        out[k] = "\n".join(lines).strip()
    return out


def drop_context_fields(context: dict[str, str], fields: str) -> dict[str, str]:
    """Remove the labelled lines named in ``fields`` (comma-separated) from every thread.

    A line is removed when it starts with ``<field>:``. Non-string values pass through.
    """
    prefixes = tuple(f.strip() + ":" for f in fields.split(",") if f.strip())
    return {k: ("\n".join(line for line in v.splitlines() if not line.strip().startswith(prefixes))
                if isinstance(v, str) else v)
            for k, v in context.items()}


def contextualise(labels: Iterable[str], context: dict[str, str],
                  missing_counter: dict[str, int] | None = None) -> list[str]:
    """Swap each candidate for its context block, keyed on the G2P id in the label.

    A label whose id is absent from ``context`` is kept as written and counted under
    ``missing_counter["missing"]`` when a counter is given; the count is reported in
    ``run_meta.json``.
    """
    out = []
    for lab in labels:
        m = G2P_ID_RE.search(lab)
        gid = m.group(0) if m else None
        if gid in context:
            out.append(context[gid])
        else:
            out.append(lab)
            if missing_counter is not None:
                missing_counter["missing"] = missing_counter.get("missing", 0) + 1
    return out


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------
def load_panel_siblings(path: Any) -> dict[str, dict]:
    """Index an all-panel G2P export (columns ``g2p id``, ``gene symbol``, ``panel``).

    Returns ``{"gene": id -> gene symbol, "ids": gene symbol -> [ids], "panel": id -> panel}``.
    """
    g = pd.read_csv(path, dtype=str)
    g.columns = [c.strip() for c in g.columns]
    g = g.dropna(subset=["g2p id", "gene symbol"])
    return {"gene": dict(zip(g["g2p id"], g["gene symbol"])),
            "ids": g.groupby("gene symbol")["g2p id"].apply(list).to_dict(),
            "panel": dict(zip(g["g2p id"], g["panel"].fillna("")))}


def add_panel_siblings(labels: Iterable[str], siblings: dict[str, dict]) -> list[str]:
    """Append, as bare ids, every entry of the candidates' genes that is not already offered.

    Genes are visited in candidate order and their entries in export order, so the added
    ids follow the original candidates. ``contextualise`` later renders the bare ids into
    full blocks.
    """
    labels = list(labels)
    present = set(candidate_ids(labels))
    for gene in dict.fromkeys(siblings["gene"].get(i) for i in candidate_ids(labels)):
        for sid in siblings["ids"].get(gene, []):
            if sid not in present:
                labels.append(sid)
                present.add(sid)
    return labels


def label_panel(label: str, siblings: dict[str, dict]) -> str:
    """Append a ``G2P Panel: <panel>`` line to a candidate block whose id has a known panel."""
    m = G2P_ID_RE.search(str(label))
    panel = siblings["panel"].get(m.group(0)) if m else None
    return f"{label}\nG2P Panel: {panel}" if panel else label


__all__: Sequence[str] = (
    "PROMPT_DIR", "DEFAULT_PROMPT_FILE", "load_prompt_template", "fill_placeholders",
    "render_candidates", "build_llm_prompt", "candidate_list", "candidate_ids",
    "load_context_threads", "drop_context_fields", "contextualise", "load_panel_siblings",
    "add_panel_siblings", "label_panel",
)
