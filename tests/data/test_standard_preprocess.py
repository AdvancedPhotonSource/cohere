import numpy as np
import pytest

import cohere_core.data.standard_preprocess as sp


def test_prep_basic_processing_and_save(monkeypatch):
    """
    Verify prep:
    - reads tif input
    - applies intensity threshold
    - takes sqrt
    - calls adjust_dimensions
    - calls binning when configured
    - calls array_to_good_dims
    - centers max unless disabled
    - applies shift
    - saves output to data.tif
    - returns kwargs
    """
    beam_data = np.array(
        [
            [[1.0, 4.0], [9.0, 16.0]],
            [[25.0, 36.0], [49.0, 64.0]],
        ]
    )

    saved = {}

    def fake_read_tif(path):
        assert path == "input/file.tif"
        return beam_data.copy()

    def fake_adjust_dimensions(data, pairs):
        assert pairs == [[0, 0], [0, 0], [0, 0]]
        return data

    def fake_binning(data, bins):
        assert bins == [1, 1, 1]
        return data

    def fake_array_to_good_dims(data, pkg):
        assert pkg == "np"
        return data

    def fake_center_max(data):
        return data, [0, 0, 0]

    def fake_join(data_dir, filename):
        return f"{data_dir}/{filename}"

    def fake_save_tif(data, path):
        saved["data"] = data.copy()
        saved["path"] = path

    monkeypatch.setattr(sp.ut, "read_tif", fake_read_tif)
    monkeypatch.setattr(sp.ut, "adjust_dimensions", fake_adjust_dimensions)
    monkeypatch.setattr(sp.ut, "binning", fake_binning)
    monkeypatch.setattr(sp.ut, "array_to_good_dims", fake_array_to_good_dims)
    monkeypatch.setattr(sp.ut, "center_max", fake_center_max)
    monkeypatch.setattr(sp.ut, "join", fake_join)
    monkeypatch.setattr(sp.ut, "save_tif", fake_save_tif)

    kwargs = {
        "data_dir": "out",
        "intensity_threshold": 10.0,
        "crop_pad": [0, 0, 0, 0, 0, 0],
        "binning": [1, 1, 1],
        "shift": [1, 0, 0],
        "pkg": "np",
    }

    result = sp.prep("input/file.tif", **kwargs)

    thresholded = np.where(beam_data <= 10.0, 0.0, beam_data)
    expected = np.sqrt(thresholded)
    expected = np.roll(expected, [1, 0, 0], axis=(0, 1, 2))

    assert result["intensity_threshold"] == 10.0
    assert saved["path"] == "out/data.tif"
    np.testing.assert_allclose(saved["data"], expected)


def test_prep_uses_remove_aliens_when_alien_alg_present(monkeypatch):
    beam_data = np.ones((2, 2, 2), dtype=float)
    alien_removed = np.full((2, 2, 2), 25.0)

    called = {}

    monkeypatch.setattr(sp.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(
        sp.at,
        "remove_aliens",
        lambda data, kwargs, data_dir: alien_removed.copy(),
    )
    monkeypatch.setattr(sp.ut, "adjust_dimensions", lambda data, pairs: data)
    monkeypatch.setattr(sp.ut, "array_to_good_dims", lambda data, pkg: data)
    monkeypatch.setattr(sp.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(sp.ut, "join", lambda d, f: f"{d}/{f}")

    def fake_save_tif(data, path):
        called["data"] = data.copy()
        called["path"] = path

    monkeypatch.setattr(sp.ut, "save_tif", fake_save_tif)

    sp.prep(
        "input/file.tif",
        data_dir="out",
        alien_alg="block_aliens",
        intensity_threshold=10.0,
    )

    expected = np.sqrt(alien_removed)
    np.testing.assert_allclose(called["data"], expected)
    assert called["path"] == "out/data.tif"


def test_prep_returns_none_when_intensity_threshold_missing(monkeypatch, capsys):
    monkeypatch.setattr(sp.ut, "read_tif", lambda path: np.ones((2, 2, 2)))

    result = sp.prep("input/file.tif", data_dir="out")

    captured = capsys.readouterr()
    assert result is None
    assert "define intensity threshold or set to auto, exiting." in captured.out


def test_prep_auto_intensity_threshold_sets_value_and_returns_it(monkeypatch):
    beam_data = np.array([[[10.0, 20.0], [30.0, 40.0]]])

    saved = {}

    monkeypatch.setattr(sp.ut, "read_tif", lambda path: beam_data.copy())
    monkeypatch.setattr(sp.ut, "adjust_dimensions", lambda data, pairs: data)
    monkeypatch.setattr(sp.ut, "array_to_good_dims", lambda data, pkg: data)
    monkeypatch.setattr(sp.ut, "center_max", lambda data: (data, [0]))
    monkeypatch.setattr(sp.ut, "join", lambda d, f: f"{d}/{f}")
    monkeypatch.setattr(
        sp.ut,
        "save_tif",
        lambda data, path: saved.update({"data": data.copy(), "path": path}),
    )

    result = sp.prep(
        "input/file.tif",
        data_dir="out",
        auto_intensity_threshold=True,
    )

    auto_threshold = max(2.0, 0.141 * beam_data[np.nonzero(beam_data)].mean().item() - 3.062)
    expected = np.sqrt(np.where(beam_data <= auto_threshold, 0.0, beam_data))

    assert result["intensity_threshold"] == auto_threshold
    np.testing.assert_allclose(saved["data"], expected)


def test_prep_raises_if_input_is_data_tif_and_data_dir_not_provided(monkeypatch):
    monkeypatch.setattr(sp.ut, "read_tif", lambda path: np.ones((2, 2, 2)))

    with pytest.raises(AttributeError, match="Define data_dir or rename the data file"):
        sp.prep("some/path/data.tif", intensity_threshold=1.0)


def test_prep_returns_none_when_adjust_dimensions_fails(monkeypatch, capsys):
    monkeypatch.setattr(sp.ut, "read_tif", lambda path: np.ones((2, 2, 2)))
    monkeypatch.setattr(sp.ut, "adjust_dimensions", lambda data, pairs: None)

    result = sp.prep(
        "input/file.tif",
        data_dir="out",
        intensity_threshold=1.0,
        crop_pad=[0, 0, 0, 0, 0, 0],
    )

    captured = capsys.readouterr()
    assert result is None
    assert 'check "crop_pad" configuration, exiting' in captured.out


def test_prep_skips_center_max_when_no_center_max_true(monkeypatch):
    beam_data = np.array([[[16.0]]])
    saved = {}
    center_called = {"called": False}

    monkeypatch.setattr(sp.ut, "read_tif", lambda path: beam_data.copy())
    monkeypatch.setattr(sp.ut, "adjust_dimensions", lambda data, pairs: data)
    monkeypatch.setattr(sp.ut, "array_to_good_dims", lambda data, pkg: data)
    monkeypatch.setattr(sp.ut, "join", lambda d, f: f"{d}/{f}")
    monkeypatch.setattr(
        sp.ut,
        "save_tif",
        lambda data, path: saved.update({"data": data.copy(), "path": path}),
    )

    def fake_center_max(data):
        center_called["called"] = True
        return data, [0]

    monkeypatch.setattr(sp.ut, "center_max", fake_center_max)

    sp.prep(
        "input/file.tif",
        data_dir="out",
        intensity_threshold=1.0,
        no_center_max=True,
    )

    assert center_called["called"] is False
    np.testing.assert_allclose(saved["data"], np.array([[[4.0]]]))

# What these tests cover
# These tests validate the main branches of prep:
#
# normal preprocessing flow
# alien removal path via alien_alg
# missing intensity_threshold
# auto_intensity_threshold=True
# protection against overwriting data.tif
# invalid crop_pad / failed adjust_dimensions
# no_center_max=True
# Notes
# A few implementation details influenced the tests:
#
# prep imports utilities as module aliases:
#
# cohere_core.utilities.utils as ut
# cohere_core.data.alien_tools as at
# So patching is done via:
#
# monkeypatch.setattr(sp.ut, ...)
# monkeypatch.setattr(sp.at, ...)
# The function mutates kwargs in some cases, especially:
#
# adding padded binning values
# inserting intensity_threshold when auto-threshold is used
# The function saves output through ut.save_tif, so the tests capture the final array there.
#
# If you want, I can also provide:
#
# a smaller minimal test suite, or
# a fully mocked version using unittest.mock.patch instead of monkeypatch