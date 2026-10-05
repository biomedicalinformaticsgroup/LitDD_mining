"""Unit tests for the CPU helpers of the screen stage (`litdd/pipeline/screen_io.py`)."""
from __future__ import annotations

import math

import pyarrow as pa
import pytest

from litdd.pipeline.screen_io import (
    get_output_schema,
    make_tiab,
    output_schema_from,
    safe_pubdate_gt_1980,
    select_shard,
    table_from_batch_with_schema,
)


@pytest.mark.parametrize("record, expected", [
    ({"languages": "eng", "pubdate": 1981}, True),
    ({"languages": "eng", "pubdate": "1981"}, True),
    ({"languages": "eng", "pubdate": 1980}, False),
    ({"languages": "eng", "pubdate": 1979}, False),
    ({"languages": "eng", "pubdate": 0}, False),
    ({"languages": "eng", "pubdate": None}, False),
    ({"languages": "eng"}, False),
    ({"languages": "eng", "pubdate": "n/a"}, False),
    ({"languages": "eng", "pubdate": "2001-05"}, False),
    ({"languages": "eng;spa", "pubdate": 2001}, False),
    ({"languages": "spa", "pubdate": 2001}, False),
    ({"languages": None, "pubdate": 2001}, False),
    ({"pubdate": 2001}, False),
])
def test_eligibility_requires_english_and_a_year_after_1980(record, expected):
    """The eligibility predicate accepts exactly English records with an integer year above 1980."""
    assert safe_pubdate_gt_1980(record) is expected


def test_make_tiab_joins_title_and_abstract_with_one_space():
    """`tiab` is title and abstract joined by one space, with None treated as empty."""
    assert make_tiab({"title": "A title.", "abstract": "An abstract."})["tiab"] == "A title. An abstract."
    assert make_tiab({"title": "Only a title.", "abstract": None})["tiab"] == "Only a title."
    assert make_tiab({"title": None, "abstract": "Only an abstract."})["tiab"] == "Only an abstract."
    assert make_tiab({})["tiab"] == ""


def test_make_tiab_keeps_the_other_fields():
    """`make_tiab` adds `tiab` in place and leaves every other field untouched."""
    row = {"pmid": "1", "title": "T", "abstract": "A", "languages": "eng"}
    out = make_tiab(row)
    assert out is row
    assert out["pmid"] == "1" and out["languages"] == "eng"


def test_output_schema_appends_the_three_screen_fields_once():
    """The output schema is the input schema plus tiab, bert_predict and bert_score."""
    base = pa.schema([("pmid", pa.string()), ("title", pa.string()), ("abstract", pa.string())])
    out = output_schema_from(base)
    assert out.names == ["pmid", "title", "abstract", "tiab", "bert_predict", "bert_score"]
    assert out.field("bert_predict").type == pa.int64()
    assert out.field("bert_score").type == pa.float32()
    # Fields already present are not duplicated.
    assert output_schema_from(out).names == out.names


def test_get_output_schema_reads_a_parquet_file(tmp_path):
    """`get_output_schema` reads the parquet schema from disk and extends it."""
    import pyarrow.parquet as pq

    path = tmp_path / "in.parquet"
    pq.write_table(pa.table({"pmid": ["1"], "title": ["t"]}), path)
    assert get_output_schema(str(path)).names == ["pmid", "title", "tiab", "bert_predict", "bert_score"]


def _batch():
    return {
        "pmid": ["1", "2"],
        "title": ["t1", "t2"],
        "abstract": ["a1", None],
        "tiab": ["t1 a1", "t2"],
    }


def _schema():
    return output_schema_from(pa.schema([
        ("pmid", pa.string()), ("title", pa.string()), ("abstract", pa.string()),
        ("journal", pa.string()),
    ]))


def test_table_from_batch_with_scores():
    """Predictions and scores fill their columns; a schema column absent from the batch is null."""
    table = table_from_batch_with_schema(_batch(), _schema(), preds=[1, 0], scores=[0.9, 0.1])
    assert table.schema == _schema()
    assert table.num_rows == 2
    assert table.column("bert_predict").to_pylist() == [1, 0]
    assert table.column("bert_score").to_pylist() == pytest.approx([0.9, 0.1])
    assert table.column("tiab").to_pylist() == ["t1 a1", "t2"]
    assert table.column("abstract").to_pylist() == ["a1", None]
    assert table.column("journal").null_count == 2


def test_table_from_batch_without_scores_has_null_scores():
    """Without scores the bert_score column is all null and NaN scores are preserved as NaN."""
    table = table_from_batch_with_schema(_batch(), _schema(), preds=[1, -1])
    assert table.column("bert_score").null_count == 2
    with_nan = table_from_batch_with_schema(_batch(), _schema(), preds=[1, -1],
                                            scores=[0.5, float("nan")])
    scores = with_nan.column("bert_score").to_pylist()
    assert scores[0] == pytest.approx(0.5) and math.isnan(scores[1])


def test_table_from_empty_batch_has_no_rows():
    """An empty batch with no predictions yields an empty table under the same schema."""
    table = table_from_batch_with_schema({}, _schema(), preds=[])
    assert table.num_rows == 0
    assert table.schema == _schema()


def test_select_shard_splits_round_robin():
    """Shard i of n receives every n-th file starting at position i; n <= 1 keeps all files."""
    files = [f"f{i}" for i in range(7)]
    assert select_shard(files, 0, 3) == ["f0", "f3", "f6"]
    assert select_shard(files, 1, 3) == ["f1", "f4"]
    assert select_shard(files, 2, 3) == ["f2", "f5"]
    assert select_shard(files, 0, 1) == files
    assert select_shard(files, 0, 0) == files
    assert sorted(f for s in range(3) for f in select_shard(files, s, 3)) == files
