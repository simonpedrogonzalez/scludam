from random import shuffle

import numpy as np
import pytest

from scludam.utils import Colnames, one_hot_encode

cols = [
    "col1",
    "col2",
    "col3",
    "col1_error",
    "col2_error",
    "col3_error",
    "col1_col2_corr",
    "col2_col3_corr",
    "col3_col1_corr",
]


@pytest.fixture
def no_shuffle():
    return cols


@pytest.fixture
def columns():
    cop = cols.copy()
    shuffle(cop)
    return cop


@pytest.fixture
def columns_missing_error():
    return cols + ["col4"]


@pytest.fixture
def columns_missing_corr():
    return cols + ["col4", "col4_error", "col3_col4_corr", "col4_col2_corr"]


@pytest.mark.parametrize(
    "fixture_name, filter_names, correct",
    [
        ("columns", ["col4", "col1"], ["col1"]),
        ("no_shuffle", None, ["col1", "col2", "col3"]),
        ("columns", ["col2", "col1", "col4"], ["col2", "col1"]),
    ],
)
def test_get_data_names(fixture_name, filter_names, correct, request):
    assert Colnames(request.getfixturevalue(fixture_name)).data(filter_names) == correct


@pytest.mark.parametrize(
    "fixture_name, filter_names, correct",
    [
        ("columns", "col1", (["col1_error"], False)),
        ("columns", ["col1", "col4"], (["col1_error"], False)),
        ("columns", ["col2", "col1"], (["col2_error", "col1_error"], False)),
        (
            "columns_missing_error",
            None,
            (["col1_error", "col2_error", "col3_error"], True),
        ),
    ],
)
def test_get_error_names(fixture_name, filter_names, correct, request):
    assert (
        Colnames(request.getfixturevalue(fixture_name)).error(filter_names)
        == correct[0]
    )
    assert (
        Colnames(request.getfixturevalue(fixture_name)).missing_error(filter_names)
        == correct[1]
    )


@pytest.mark.parametrize(
    "fixture_name, filter_names, correct",
    [
        ("columns", "col1", ([], False)),
        ("columns", ["col2", "col1"], (["col1_col2_corr"], False)),
        (
            "columns",
            ["col3", "col1", "col2"],
            (["col3_col1_corr", "col2_col3_corr", "col1_col2_corr"], False),
        ),
        (
            "columns_missing_corr",
            None,
            (
                [
                    "col1_col2_corr",
                    "col3_col1_corr",
                    "col2_col3_corr",
                    "col4_col2_corr",
                    "col3_col4_corr",
                ],
                True,
            ),
        ),
    ],
)
def test_get_corr_names(fixture_name, filter_names, correct, request):
    assert (
        Colnames(request.getfixturevalue(fixture_name)).corr(filter_names) == correct[0]
    )
    assert (
        Colnames(request.getfixturevalue(fixture_name)).missing_corr(filter_names)
        == correct[1]
    )


def test_one_hot_encode():
    labels = [0, 1, 1, 1, 0, 2]
    result = one_hot_encode(labels)
    assert np.allclose(
        result,
        np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    )
