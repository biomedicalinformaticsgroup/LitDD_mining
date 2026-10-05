"""Helpers shared by the evaluation scripts: id parsing, confusion-matrix metrics, Wilson
intervals, the X-linked entry equivalence rule and paired significance tests."""
from __future__ import annotations

import json
import math
import random
import re
from math import comb

import pandas as pd

G2P_ID_RE = re.compile(r"G2P\d+")
NO_MATCH = "NO MATCH"


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def g2p_ids_regex(cell) -> set[str]:
    """Every ``G2P\\d+`` token in a cell; ``NO MATCH``, None, NaN and '' give the empty set."""
    if _is_missing(cell):
        return set()
    s = str(cell).strip()
    if not s or s.upper() == NO_MATCH or s == "nan":
        return set()
    return set(G2P_ID_RE.findall(s))


def g2p_ids_split(cell) -> set[str]:
    """Ids in a ``;``-joined cell, taken verbatim; ``NO MATCH``, None, NaN and '' give the empty set."""
    if _is_missing(cell):
        return set()
    s = str(cell).strip()
    if not s or s.upper() == NO_MATCH or s == "nan":
        return set()
    return {x.strip() for x in s.split(";") if x.strip() and x.strip() != "nan"}


def candidate_ids_from_row(cell) -> list[str]:
    """G2P ids in a ``candidates`` cell (list, ndarray, None or NaN), in stored order."""
    if _is_missing(cell):
        return []
    out = []
    for item in list(cell):
        m = G2P_ID_RE.search(str(item))
        if m:
            out.append(m.group(0))
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials; NaNs when ``n`` is zero."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def prf(tp: int, fp: int, fn: int, empty: float = 0.0) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts; ``empty`` is returned where a denominator is zero."""
    p = tp / (tp + fp) if tp + fp else empty
    r = tp / (tp + fn) if tp + fn else empty
    f = 2 * p * r / (p + r) if (p + r) and not (math.isnan(p) or math.isnan(r)) else empty
    return p, r, f


def x_equivalence_map(g2p_csv: str) -> dict[str, str]:
    """Map each member of a group of X-linked entries to a canonical id.

    Entries of one gene with the same disease name whose allelic requirement differs only
    between monoallelic_X_heterozygous and monoallelic_X_hemizygous are scored as the same
    entry. Every member of such a group maps to the lowest id of the group."""
    d = pd.read_csv(g2p_csv)
    x = d[d["allelic requirement"].astype(str).str.startswith("monoallelic_X")]
    canon: dict[str, str] = {}
    for _, grp in x.groupby(["gene symbol", "disease name"]):
        ids = sorted(grp["g2p id"].astype(str))
        if len(ids) > 1:
            for i in ids:
                canon[i] = ids[0]
    return canon


def read_provenance_json(path: str) -> dict:
    """A JSON object without its ``__``-prefixed provenance keys."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not str(k).startswith("__")}


def f1_binary(labels, preds) -> float:
    tp = sum(1 for lab, p in zip(labels, preds) if lab == 1 and p == 1)
    fp = sum(1 for lab, p in zip(labels, preds) if lab == 0 and p == 1)
    fn = sum(1 for lab, p in zip(labels, preds) if lab == 1 and p == 0)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def mcnemar_exact(labels, a, b) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired predictions.

    Returns the number of items only ``a`` gets right, the number only ``b`` gets right,
    and the two-sided binomial p-value."""
    b_only = sum(1 for lab, x, y in zip(labels, a, b) if (x == lab) and (y != lab))
    c_only = sum(1 for lab, x, y in zip(labels, a, b) if (x != lab) and (y == lab))
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    k = min(b_only, c_only)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b_only, c_only, p


def bootstrap_f1_diff(labels, a, b, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """F1(a) - F1(b) with a 95% percentile bootstrap interval over resampled items."""
    rng = random.Random(seed)
    n = len(labels)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        diffs.append(f1_binary(la, [a[i] for i in idx]) - f1_binary(la, [b[i] for i in idx]))
    diffs.sort()
    return (f1_binary(labels, a) - f1_binary(labels, b),
            diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1])
