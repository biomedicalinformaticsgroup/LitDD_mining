"""split_external: held-out, dropped and trainable external papers by gene bucket."""
import pandas as pd

from litdd.training.build_screen_training_set import split_external
from litdd.training.screen_common import gene_fold


def _gene_in_bucket(want_zero: bool) -> str:
    return next(g for g in (f"G{i}" for i in range(1000)) if (gene_fold(g, 10) == 0) == want_zero)


def test_split_external_by_first_gene_bucket():
    held, train = _gene_in_bucket(True), _gene_in_bucket(False)
    ext = pd.DataFrame({
        "pmid": ["1", "2", "3", "3"],
        "gene": [held, train, train, held],      # PMID 3 is bucketed on its first row only
        "tiab": ["a", "b", "c", "c"],
        "g2p_id": ["X1", "X2", "X3", "X4"],
        "source": ["premined", "hpoa", "clingen", "clingen"],
    })
    trainable, heldout = split_external(ext)
    assert list(heldout["pmid"]) == ["1"]
    assert list(trainable["pmid"]) == ["2", "3"]
    assert trainable.set_index("pmid").loc["3", "g2p_id"] == "X3"
