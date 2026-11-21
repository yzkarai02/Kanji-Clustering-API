# Copyright (c) 2026 yudukikun5120

"""Unit tests for Kanji clustering API."""

import hashlib
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from .affinities_detection import get_affinities
from .clustering import (
    create_fitted_estimator,
    extract_feature,
    kanji_group,
    store_estimator,
)
from .estimator_store import (
    EstimatorIntegrityError,
    load_estimator,
    store_estimator_file,
)
from .main import affinities, app
from .preprocessing import ndarray_of

# Constants for expected values
JIS_LEVEL_1_COUNT = 2965
JIS_LEVEL_2_COUNT = 3390
EXPECTED_AFFINITY_COUNT = 3
IMAGE_SHAPE = (64, 64, 3)
SMALL_GROUP_SIZE = 4
PAIR_COUNT = 2


def test_kanji_jis_daiichisuijun() -> None:
    """Test that JIS Level 1 kanji group has the expected number of characters."""
    result_length = len(kanji_group("jis_level_1"))
    if result_length != JIS_LEVEL_1_COUNT:
        msg = f"Expected {JIS_LEVEL_1_COUNT} characters, got {result_length}"
        raise AssertionError(msg)


@patch("kanji_clustering_api.clustering.store_estimator_file")
@patch("kanji_clustering_api.clustering.create_fitted_estimator")
def test_store_estimator(
    mock_create_fitted_estimator: MagicMock,
    mock_store_file: MagicMock,
) -> None:
    """Test that estimators can be created and stored for both JIS levels."""
    # Mock the create_fitted_estimator to return dummy data
    mock_estimator = MagicMock()
    mock_df = pd.DataFrame({"character": ["あ", "い"], "label": [0, 1]})
    mock_create_fitted_estimator.return_value = (mock_estimator, mock_df)

    # Test storing estimator
    store_estimator("jis_level_1")

    # Verify the function was called
    mock_create_fitted_estimator.assert_called_once_with("jis_level_1")
    mock_store_file.assert_called_once()


@patch("kanji_clustering_api.affinities_detection.load_estimator")
@patch("kanji_clustering_api.affinities_detection.ndarray_of")
def test_affinities(
    mock_ndarray_of: MagicMock,
    mock_load_estimator: MagicMock,
) -> None:
    """Test that affinities can be retrieved for sample characters."""
    # Mock the estimator and dataframe
    mock_estimator = MagicMock()
    mock_estimator.predict.return_value = np.array([0])  # Predict label 0

    # Create a mock dataframe with some test data
    mock_df = pd.DataFrame(
        {
            "character": ["蟻", "蟷", "蜘", "蛛", "虫"],
            "label": [0, 0, 0, 1, 1],
        },
    )

    mock_load_estimator.return_value = (mock_estimator, mock_df)

    # Mock ndarray_of to return a dummy array
    mock_ndarray_of.return_value = np.zeros((64, 64, 3))

    # Test getting affinities
    result = get_affinities("蟻", "jis_level_1")

    # Verify the result contains characters with the same label
    if len(result) != EXPECTED_AFFINITY_COUNT:
        msg = f"Expected {EXPECTED_AFFINITY_COUNT} characters, got {len(result)}"
        raise AssertionError(msg)
    if "蟻" not in result:
        msg = "Expected '蟻' in result"
        raise AssertionError(msg)
    if "蟷" not in result:
        msg = "Expected '蟷' in result"
        raise AssertionError(msg)
    if "蜘" not in result:
        msg = "Expected '蜘' in result"
        raise AssertionError(msg)


@patch("kanji_clustering_api.main.get_affinities")
def test_affinities_endpoint_deduplicates_repeated_sets(
    mock_get_affinities: MagicMock,
) -> None:
    """Repeating a set name in `sets` must not call get_affinities repeatedly.

    Regression test: a client could previously repeat "jis_level_1" up to
    ~1,358 times within a single HTTP request (bounded by h11's 16 KiB
    request-line/header limit) and force an uncached pickle load plus model
    inference for each repetition, multiplying the cost of one request.
    """
    mock_get_affinities.return_value = np.array(["蟻"])

    affinities(character="蟻", sets="jis_level_1 jis_level_1 jis_level_1")

    mock_get_affinities.assert_called_once_with("蟻", "jis_level_1")


def test_load_estimator_accepts_pinned_files() -> None:
    """The committed estimator files must match the committed manifest."""
    for kanji_set in ("jis_level_1", "jis_level_2"):
        estimator, df = load_estimator(kanji_set)
        assert hasattr(estimator, "predict")  # noqa: S101
        assert {"character", "label"} <= set(df.columns)  # noqa: S101


def test_load_estimator_rejects_tampered_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swapped pickle (even a valid one) must never be deserialized."""
    estimator_dir = tmp_path / "estimator"
    estimator_dir.mkdir()
    good = pickle.dumps(("estimator", "df"))
    (estimator_dir / "x.pkl").write_bytes(good)
    (estimator_dir / "SHA256SUMS").write_text(
        f"{hashlib.sha256(good).hexdigest()}  x.pkl\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "kanji_clustering_api.estimator_store.ESTIMATOR_DIR",
        estimator_dir,
    )
    monkeypatch.setattr(
        "kanji_clustering_api.estimator_store.MANIFEST_PATH",
        estimator_dir / "SHA256SUMS",
    )
    assert load_estimator("x") == ("estimator", "df")  # noqa: S101

    (estimator_dir / "x.pkl").write_bytes(pickle.dumps(("evil", "df")))
    with pytest.raises(EstimatorIntegrityError):
        load_estimator("x")
    (estimator_dir / "unpinned.pkl").write_bytes(good)
    with pytest.raises(EstimatorIntegrityError):
        load_estimator("unpinned")


def test_kanji_jis_dainisuijun() -> None:
    """Test that JIS Level 2 kanji group has the expected number of characters."""
    result_length = len(kanji_group("jis_level_2"))
    if result_length != JIS_LEVEL_2_COUNT:
        msg = f"Expected {JIS_LEVEL_2_COUNT} characters, got {result_length}"
        raise AssertionError(msg)


def test_ndarray_of_renders_character() -> None:
    """A character renders deterministically to a 64x64 RGB array."""
    result = ndarray_of("あ")

    assert result.shape == IMAGE_SHAPE  # noqa: S101
    assert result.dtype == np.uint8  # noqa: S101
    assert int(result.sum()) > 0  # noqa: S101
    assert np.array_equal(result, ndarray_of("あ"))  # noqa: S101
    assert not np.array_equal(result, ndarray_of("い"))  # noqa: S101


@patch("kanji_clustering_api.clustering.ndarray_of")
@patch("kanji_clustering_api.clustering.kanji_group")
def test_extract_feature_flattens_each_character(
    mock_kanji_group: MagicMock,
    mock_ndarray_of: MagicMock,
) -> None:
    """Each character's image is flattened into one feature row."""
    mock_kanji_group.return_value = ["あ", "い"]
    mock_ndarray_of.side_effect = [np.ones((2, 2)), np.zeros((2, 2))]

    result = extract_feature("jis_level_1")

    assert result.shape == (2, SMALL_GROUP_SIZE)  # noqa: S101
    assert np.array_equal(result[0], np.ones(SMALL_GROUP_SIZE))  # noqa: S101
    assert np.array_equal(result[1], np.zeros(SMALL_GROUP_SIZE))  # noqa: S101


@patch("kanji_clustering_api.clustering.kanji_group")
@patch("kanji_clustering_api.clustering.extract_feature")
def test_create_fitted_estimator(
    mock_extract_feature: MagicMock,
    mock_kanji_group: MagicMock,
) -> None:
    """Fitting returns a labelled estimator and a character/label frame."""
    characters = ["あ", "い", "う", "え"]
    mock_kanji_group.return_value = characters
    mock_extract_feature.return_value = np.array(
        [[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]],
    )

    estimator, df = create_fitted_estimator("jis_level_1", n_clusters=2)

    assert list(df.columns) == ["character", "label"]  # noqa: S101
    assert df["character"].tolist() == characters  # noqa: S101
    assert len(estimator.labels_) == SMALL_GROUP_SIZE  # noqa: S101
    assert set(estimator.labels_.tolist()) == {0, 1}  # noqa: S101
    # Well-separated pairs must land in different clusters from each other.
    assert df["label"].iloc[0] == df["label"].iloc[1]  # noqa: S101
    assert df["label"].iloc[2] == df["label"].iloc[3]  # noqa: S101
    assert df["label"].iloc[0] != df["label"].iloc[2]  # noqa: S101


def test_store_estimator_file_roundtrip_and_manifest_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stored objects load back, and the manifest tracks every file."""
    estimator_dir = tmp_path / "estimator"
    estimator_dir.mkdir()
    manifest_path = estimator_dir / "SHA256SUMS"
    monkeypatch.setattr(
        "kanji_clustering_api.estimator_store.ESTIMATOR_DIR",
        estimator_dir,
    )
    monkeypatch.setattr(
        "kanji_clustering_api.estimator_store.MANIFEST_PATH",
        manifest_path,
    )

    # First store: no manifest exists yet, so one is created from scratch.
    store_estimator_file("alpha", ("estimator", "df"))
    assert load_estimator("alpha") == ("estimator", "df")  # noqa: S101
    digest = hashlib.sha256((estimator_dir / "alpha.pkl").read_bytes()).hexdigest()
    assert manifest_path.read_text(encoding="utf-8") == f"{digest}  alpha.pkl\n"  # noqa: S101

    # A blank line in the manifest must be ignored when it is re-read, and
    # storing a second set must keep the existing entry, sorted by name.
    with manifest_path.open("a", encoding="utf-8") as manifest:
        manifest.write("\n")
    store_estimator_file("beta", ("other", "df"))
    assert load_estimator("beta") == ("other", "df")  # noqa: S101
    names = [
        line.split(maxsplit=1)[1]
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert names == ["alpha.pkl", "beta.pkl"]  # noqa: S101

    # Overwriting a set replaces its entry instead of duplicating it.
    store_estimator_file("alpha", ("new", "df"))
    assert load_estimator("alpha") == ("new", "df")  # noqa: S101
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == PAIR_COUNT  # noqa: S101


@patch("kanji_clustering_api.main.get_affinities")
def test_affinities_endpoint_combines_valid_sets_and_skips_invalid(
    mock_get_affinities: MagicMock,
) -> None:
    """Valid sets are combined in order; unknown set names are ignored."""
    mock_get_affinities.side_effect = [np.array(["蟻"]), np.array(["蛛"])]

    result = affinities(character="蟻", sets="jis_level_1 unknown jis_level_2")

    assert result == {"character": "蟻", "affinities": ["蟻", "蛛"]}  # noqa: S101
    assert mock_get_affinities.call_count == PAIR_COUNT  # noqa: S101


@patch("kanji_clustering_api.main.get_affinities")
def test_affinities_endpoint_with_no_valid_sets(
    mock_get_affinities: MagicMock,
) -> None:
    """A request naming no valid set returns an empty affinity list."""
    result = affinities(character="蟻", sets="unknown")

    assert result == {"character": "蟻", "affinities": []}  # noqa: S101
    mock_get_affinities.assert_not_called()


def test_affinities_character_limited_to_one_code_point() -> None:
    """`character` must be declared as exactly one code point.

    The feature extractor renders it onto a fixed 64x64 canvas, so a longer
    string (or a newline-separated one) only adds rendering work.
    """
    parameters = app.openapi()["paths"]["/affinities"]["get"]["parameters"]
    character = next(p for p in parameters if p["name"] == "character")

    assert character["schema"]["minLength"] == 1  # noqa: S101
    assert character["schema"]["maxLength"] == 1  # noqa: S101
