import os
import sys
import types
import importlib
import numpy as np
import pytest


@pytest.fixture
def standard_preprocess_module(monkeypatch):
    """
    Build fake cohere_core package structure before importing standard_preprocess.
    """

    # -------------------------
    # fake utils module
    # -------------------------
    fake_utils = types.ModuleType("cohere_core.utilities.utils")

    def read_tif(path):
        raise NotImplementedError("read_tif should be monkeypatched in each test")

    def save_tif(arr, path):
        pass

    def join(*args):
        return os.path.join(*args)

    def adjust_dimensions(data, pairs, next_fast_len, pkg):
        return data

    def center_max(data):
        return data, [0, 0, 0]

    def binning(data, bins):
        return data

    fake_utils.read_tif = read_tif
    fake_utils.save_tif = save_tif
    fake_utils.join = join
    fake_utils.adjust_dimensions = adjust_dimensions
    fake_utils.center_max = center_max
    fake_utils.binning = binning

    # -------------------------
    # fake alien_tools module
    # -------------------------
    fake_alien_tools = types.ModuleType("cohere_core.data.alien_tools")

    def remove_aliens(data, kwargs, data_dir=None):
        return data

    fake_alien_tools.remove_aliens = remove_aliens

    # -------------------------
    # fake package hierarchy
    # -------------------------
    cohere_core = types.ModuleType("cohere_core")
    data_pkg = types.ModuleType("cohere_core.data")
    utilities_pkg = types.ModuleType("cohere_core.utilities")

    data_pkg.alien_tools = fake_alien_tools
    utilities_pkg.utils = fake_utils
    cohere_core.data = data_pkg
    cohere_core.utilities = utilities_pkg

    monkeypatch.setitem(sys.modules, "cohere_core", cohere_core)
    monkeypatch.setitem(sys.modules, "cohere_core.data", data_pkg)
    monkeypatch.setitem(sys.modules, "cohere_core.data.alien_tools", fake_alien_tools)
    monkeypatch.setitem(sys.modules, "cohere_core.utilities", utilities_pkg)
    monkeypatch.setitem(sys.modules, "cohere_core.utilities.utils", fake_utils)

    # import/reload module under test
    if "standard_preprocess" in sys.modules:
        mod = importlib.reload(sys.modules["standard_preprocess"])
    else:
        mod = importlib.import_module("standard_preprocess")

    return mod


def test_prep_raises_if_input_file_is_data_tif_without_data_dir(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: np.ones((2, 2, 2), dtype=float))

    with pytest.raises(AttributeError, match="Define data_dir or rename the data file"):
        mod.prep("/tmp/data.tif", intensity_threshold=1.0)


def test_prep_returns_none_if_no_intensity_threshold_and_not_auto(standard_preprocess_module, monkeypatch, capsys):
    mod = standard_preprocess_module

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: np.ones((2, 2, 2), dtype=float))

    result = mod.prep("/tmp/input.tif")
    captured = capsys.readouterr()

    assert result is None
    assert "define intensity threshold or set to auto, exiting." in captured.out


def test_prep_calls_remove_aliens_when_alien_alg_configured(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float) * 9
    saved = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)

    def fake_remove_aliens(data, kwargs, data_dir=None):
        saved["called"] = True
        saved["data_dir"] = data_dir
        return data * 2

    monkeypatch.setattr(mod.at, "remove_aliens", fake_remove_aliens)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: saved.update({"saved_array": arr, "saved_path": path}))

    result = mod.prep(
        "/tmp/input.tif",
        data_dir="/tmp/out",
        alien_alg="block_aliens",
        intensity_threshold=1.0,
    )

    assert saved["called"] is True
    assert saved["data_dir"] == "/tmp/out"
    assert saved["saved_path"] == os.path.join("/tmp/out", "data.tif")
    assert isinstance(result, dict)


def test_prep_applies_intensity_threshold_and_sqrt(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.array(
        [[[1.0, 4.0], [9.0, 16.0]],
         [[0.0, 25.0], [36.0, 2.0]]]
    )
    captured = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: captured.update({"arr": arr, "path": path}))

    mod.prep("/tmp/input.tif", data_dir="/tmp/out", intensity_threshold=4.0)

    expected_thresholded = np.where(beam_data <= 4.0, 0.0, beam_data)
    expected = np.sqrt(expected_thresholded)

    assert np.array_equal(captured["arr"], expected)
    assert captured["path"] == os.path.join("/tmp/out", "data.tif")


def test_prep_auto_intensity_threshold_sets_value_in_returned_kwargs(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.array(
        [[[10.0, 20.0], [30.0, 40.0]],
         [[50.0, 60.0], [70.0, 80.0]]]
    )

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: None)

    kwargs = {
        "data_dir": "/tmp/out",
        "auto_intensity_threshold": True,
    }

    result = mod.prep("/tmp/input.tif", **kwargs)

    assert "intensity_threshold" in result
    assert result["intensity_threshold"] >= 2.0


def test_prep_extends_crop_pad_before_adjust_dimensions(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float)
    captured = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)

    def fake_adjust_dimensions(data, pairs, next_fast_len, pkg):
        captured["pairs"] = pairs
        captured["next_fast_len"] = next_fast_len
        captured["pkg"] = pkg
        return data

    monkeypatch.setattr(mod.ut, "adjust_dimensions", fake_adjust_dimensions)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: None)

    mod.prep(
        "/tmp/input.tif",
        data_dir="/tmp/out",
        intensity_threshold=1.0,
        crop_pad=[1, 2, 3],
        next_fast_len=False,
        pkg="np",
    )

    assert captured["pairs"] == [[1, 2], [3, 0], [0, 0]]
    assert captured["next_fast_len"] is False
    assert captured["pkg"] == "np"


def test_prep_returns_none_when_adjust_dimensions_returns_none(standard_preprocess_module, monkeypatch, capsys):
    mod = standard_preprocess_module

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: np.ones((2, 2, 2), dtype=float))
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: None)

    result = mod.prep("/tmp/input.tif", data_dir="/tmp/out", intensity_threshold=1.0)
    captured = capsys.readouterr()

    assert result is None
    assert 'check "crop_pad" configuration' in captured.out


def test_prep_skips_centering_when_no_center_max_true(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float)
    called = {"center_max": False}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)

    def fake_center_max(data):
        called["center_max"] = True
        return data, [0, 0, 0]

    monkeypatch.setattr(mod.ut, "center_max", fake_center_max)
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: None)

    mod.prep(
        "/tmp/input.tif",
        data_dir="/tmp/out",
        intensity_threshold=1.0,
        no_center_max=True,
    )

    assert called["center_max"] is False


def test_prep_applies_shift(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.zeros((3, 3, 3), dtype=float)
    beam_data[0, 0, 0] = 9.0
    captured = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: captured.update({"arr": arr}))

    mod.prep(
        "/tmp/input.tif",
        data_dir="/tmp/out",
        intensity_threshold=0.0,
        shift=[1, 1, 1],
    )

    expected = np.sqrt(beam_data)
    expected = np.roll(expected, [1, 1, 1], (0, 1, 2))

    assert np.array_equal(captured["arr"], expected)


def test_prep_reads_and_saves_mask_when_present(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float) * 9
    mask_data = np.ones((2, 2, 2), dtype=float)
    saved = []

    def fake_read_tif(path):
        if path.endswith("input.tif"):
            return beam_data
        if path.endswith("mask.tif"):
            return mask_data
        raise FileNotFoundError(path)

    monkeypatch.setattr(mod.ut, "read_tif", fake_read_tif)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [1, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: saved.append((arr.copy(), path)))

    mod.prep("/tmp/input.tif", data_dir="/tmp/out", intensity_threshold=1.0)

    saved_paths = [p for _, p in saved]
    assert os.path.join("/tmp/out", "mask.tif") in saved_paths
    assert os.path.join("/tmp/out", "data.tif") in saved_paths


def test_prep_ignores_missing_mask_file(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float) * 9
    saved = []

    def fake_read_tif(path):
        if path.endswith("input.tif"):
            return beam_data
        raise FileNotFoundError(path)

    monkeypatch.setattr(mod.ut, "read_tif", fake_read_tif)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: saved.append(path))

    mod.prep("/tmp/input.tif", data_dir="/tmp/out", intensity_threshold=1.0)

    assert os.path.join("/tmp/out", "data.tif") in saved


def test_prep_applies_binning_and_updates_kwargs(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((4, 4, 4), dtype=float) * 16
    captured = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))

    def fake_binning(data, bins):
        captured["bins"] = bins
        return data

    monkeypatch.setattr(mod.ut, "binning", fake_binning)
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: None)

    kwargs = {
        "data_dir": "/tmp/out",
        "intensity_threshold": 1.0,
        "binning": [2, 2],
    }

    result = mod.prep("/tmp/input.tif", **kwargs)

    assert captured["bins"] == [2, 2, 1]
    assert result["binning"] == [2, 2, 1]


def test_prep_raises_when_binning_fails(standard_preprocess_module, monkeypatch, capsys):
    mod = standard_preprocess_module

    beam_data = np.ones((4, 4, 4), dtype=float) * 16

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))

    def fake_binning(data, bins):
        raise ValueError("bad binning")

    monkeypatch.setattr(mod.ut, "binning", fake_binning)
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: None)

    with pytest.raises(ValueError, match="bad binning"):
        mod.prep(
            "/tmp/input.tif",
            data_dir="/tmp/out",
            intensity_threshold=1.0,
            binning=[2, 2, 2],
        )

    captured = capsys.readouterr()
    assert 'check "binning" configuration' in captured.out


def test_prep_saves_to_input_directory_when_data_dir_not_given(standard_preprocess_module, monkeypatch):
    mod = standard_preprocess_module

    beam_data = np.ones((2, 2, 2), dtype=float) * 9
    saved = {}

    monkeypatch.setattr(mod.ut, "read_tif", lambda path: beam_data)
    monkeypatch.setattr(mod.ut, "adjust_dimensions", lambda data, pairs, next_fast_len, pkg: data)
    monkeypatch.setattr(mod.ut, "center_max", lambda data: (data, [0, 0, 0]))
    monkeypatch.setattr(mod.ut, "save_tif", lambda arr, path: saved.update({"path": path}))

    mod.prep("/tmp/my_input.tif", intensity_threshold=1.0)

    assert saved["path"] == os.path.join("/tmp", "data.tif")


# What these tests cover
# These tests validate:
#
# reading input tif
# handling missing intensity_threshold
# auto threshold behavior
# alien removal dispatch via at.remove_aliens
# thresholding and square root conversion
# crop_pad expansion and passing pairs to adjust_dimensions
# behavior when adjust_dimensions fails
# optional centering
# manual shifting
# mask loading/saving behavior
# binning behavior
# save path logic
# protection against overwriting data.tif
# Notes
# 1. Import path
# This assumes your file is importable as:
#
#
# import standard_preprocess
# If it lives inside a package, adjust this line in the fixture:
#
#
# mod = importlib.import_module("standard_preprocess")
# For example:
#
#
# mod = importlib.import_module("cohere_core.data.standard_preprocess")
# 2. External dependency stubbing
# Since your module imports:
#
#
# import cohere_core.data.alien_tools as at
# import cohere_core.utilities.utils as ut
# the test fixture creates fake modules before import. That avoids import errors in isolated test environments.
#
# 3. One small behavior note
# In your code, this line:
#
#
# data = np.where(data <= intensity_threshold, 0.0, data)
# means values exactly equal to the threshold are zeroed out.
#
# The test reflects that exact behavior.
#
# If you want, I can also provide:
#
# a smaller minimal test file
# a more production-style version with fixtures split cleanly
# a coverage-focused version for all branches of prep()