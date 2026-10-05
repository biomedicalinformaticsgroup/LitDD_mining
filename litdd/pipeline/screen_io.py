"""Row filtering, text assembly and output-table construction for the screen stage.

``bert_predict_vllm.py`` imports these helpers; they have no GPU dependency and are the
part of the screen that runs on CPU. The module computes:

* the corpus eligibility predicate (English-language record, publication year after 1980);
* the ``tiab`` field, title and abstract joined by one space;
* the output parquet schema, which is the input schema plus ``tiab``, ``bert_predict`` and
  ``bert_score``;
* an Arrow table for one batch of rows together with its predictions;
* the round-robin split of an input file list across shards.
"""
from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

# Rows pulled from the streaming dataset per CPU-side batch.
ROW_BATCH_SIZE = 8192
# Texts sent to vLLM per classify call.
PRED_BATCH_SIZE = 1024
# Token budget of the screen model (ModernBERT context length).
MAX_LENGTH = 8192
PARQUET_COMPRESSION = "zstd"
# Eligibility criteria of the screened corpus.
ELIGIBLE_LANGUAGE = "eng"
MIN_PUBYEAR_EXCLUSIVE = 1980

OUTPUT_FIELDS = (
    ("tiab", pa.string()),
    ("bert_predict", pa.int64()),
    ("bert_score", pa.float32()),
)


def safe_pubdate_gt_1980(x: dict[str, Any]) -> bool:
    """Return True for a record in English with an integer publication year above 1980.

    ``pubdate`` may be missing, ``None`` or non-numeric; each of these counts as an
    ineligible year. ``languages`` must equal ``"eng"`` exactly, so multilingual records
    such as ``"eng;spa"`` are excluded.
    """
    try:
        pubdate = x.get("pubdate", None)
        year = int(pubdate) if pubdate is not None else -1
    except Exception:
        year = -1
    return (x.get("languages") == ELIGIBLE_LANGUAGE) and (year > MIN_PUBYEAR_EXCLUSIVE)


def make_tiab(x: dict[str, Any]) -> dict[str, Any]:
    """Set ``x["tiab"]`` to the title and abstract joined by one space, treating None as empty."""
    title = x.get("title", "") or ""
    abstract = x.get("abstract", "") or ""
    x["tiab"] = f"{title} {abstract}".strip()
    return x


def output_schema_from(base: pa.Schema) -> pa.Schema:
    """Return ``base`` extended with the ``tiab``, ``bert_predict`` and ``bert_score`` fields."""
    fields = list(base)
    for name, typ in OUTPUT_FIELDS:
        if name not in base.names:
            fields.append(pa.field(name, typ))
    return pa.schema(fields)


def get_output_schema(parquet_path: str) -> pa.Schema:
    """Read the schema of ``parquet_path`` and extend it with the screen's output fields."""
    return output_schema_from(pq.read_schema(parquet_path))


def table_from_batch_with_schema(
    batch: dict[str, list[Any]],
    schema: pa.Schema,
    preds: list[int],
    scores: list[float] | None = None,
) -> pa.Table:
    """Build an Arrow table for one batch under ``schema``.

    ``batch`` maps column name to a list of values. ``preds`` fills ``bert_predict`` and
    ``scores`` fills ``bert_score`` (nulls when not given). Schema columns missing from the
    batch are filled with nulls; ``tiab`` defaults to empty strings.
    """
    if len(preds) > 0:
        n = len(preds)
    elif batch:
        n = len(batch[next(iter(batch))])
    else:
        n = 0

    columns = {}
    for field in schema:
        name = field.name
        if name == "bert_predict":
            arr = pa.array(preds, type=pa.int64())
        elif name == "bert_score":
            arr = pa.array(scores if scores is not None else [None] * n, type=pa.float32())
        elif name == "tiab":
            arr = pa.array(batch.get("tiab", [""] * n), type=pa.string())
        elif name in batch:
            arr = pa.array(batch[name], type=field.type)
        else:
            arr = pa.nulls(n, type=field.type)
        columns[name] = arr
    return pa.Table.from_arrays([columns[f.name] for f in schema], schema=schema)


def select_shard(files: list[str], shard: int, num_shards: int) -> list[str]:
    """Return the files at positions congruent to ``shard`` modulo ``num_shards``."""
    step = max(num_shards, 1)
    return [p for i, p in enumerate(files) if i % step == shard]
