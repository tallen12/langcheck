from langcheck.evaluation.models import ScoredEdit
from langcheck.taxonomy.kinds import ErrorKind


def test_the_key_ignores_the_kind() -> None:
    assert (
        ScoredEdit(1, 2, ("goes",), ErrorKind.SPELLING).key == ScoredEdit(1, 2, ("goes",), None).key
    )
