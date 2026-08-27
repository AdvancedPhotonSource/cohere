import os
import stat
import numpy as np
import tempfile
import pytest

import cohere_core.utilities.utils as utils


def test_join_uses_forward_slashes():
    path = utils.join("a", "b", "c")
    assert path.endswith("a/b/c") or path == "a/b/c"


def test_normalize_returns_unit_vector():
    vec = np.array([3.0, 4.0])
    out = utils.normalize(vec)
    assert np.allclose(out, np.array([0.6, 0.8]))
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_center_max_rolls_maximum_to_center():
    arr = np.zeros((5, 5))
    arr[0, 0] = 10
    centered, shift = utils.center_max(arr)

    center = tuple(np.array(arr.shape) // 2)
    assert np.unravel_index(np.argmax(centered), centered.shape) == center
    assert np.array_equal(shift, np.array([2, 2]))


def test_crop_center_crops_3d_array_around_center():
    arr = np.arange(5 * 5 * 5).reshape((5, 5, 5))
    cropped = utils.crop_center(arr, (3, 3, 3))

    expected = arr[1:4, 1:4, 1:4]
    assert np.array_equal(cropped, expected)


def test_crop_center_raises_for_ndim_gt_3():
    arr = np.zeros((2, 2, 2, 2))
    with pytest.raises(NotImplementedError):
        utils.crop_center(arr, (1, 1, 1, 1))


def test_pad_center_pads_1d_array_into_center():
    arr = np.array([1, 2, 3])
    padded = utils.pad_center(arr, (7,))
    expected = np.array([0, 0, 1, 2, 3, 0, 0])
    assert np.array_equal(padded, expected)


def test_pad_center_pads_2d_array_into_center():
    arr = np.array([[1, 2], [3, 4]])
    padded = utils.pad_center(arr, (6, 6))

    expected = np.zeros((6, 6), dtype=arr.dtype)
    expected[2:4, 2:4] = arr
    assert np.array_equal(padded, expected)


def test_adjust_dimensions_pads_array():
    arr = np.ones((2, 2, 2))
    out = utils.adjust_dimensions(
        arr,
        pads=[(1, 1), (1, 1), (1, 1)],
        next_fast_len=False
    )

    assert out.shape == (4, 4, 4)
    assert np.array_equal(out[1:3, 1:3, 1:3], arr)
    assert np.sum(out) == 8.0


def test_adjust_dimensions_crops_array():
    arr = np.arange(4 * 4 * 4).reshape((4, 4, 4))
    out = utils.adjust_dimensions(
        arr,
        pads=[(-1, -1), (-1, -1), (-1, -1)],
        next_fast_len=False
    )

    expected = arr[1:3, 1:3, 1:3]
    assert out.shape == (2, 2, 2)
    assert np.array_equal(out, expected)


def test_binning_bins_2d_array_by_summing_blocks():
    arr = np.arange(16).reshape((4, 4))
    out = utils.binning(arr, [2, 2])

    expected = np.array([
        [0 + 1 + 4 + 5, 2 + 3 + 6 + 7],
        [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]
    ])
    assert np.array_equal(out, expected)


def test_binning_trims_non_divisible_edges():
    arr = np.ones((5, 5))
    out = utils.binning(arr, [2, 2])

    assert out.shape == (2, 2)
    assert np.array_equal(out, np.full((2, 2), 4.0))


def test_threshold_by_edge_thresholds_based_on_edge_max():
    arr = np.zeros((3, 3))
    arr[0, 0] = 2
    arr[1, 1] = 3
    arr[2, 2] = 1

    out = utils.threshold_by_edge(arr)
    expected = np.zeros((3, 3))
    expected[1, 1] = 1

    assert np.array_equal(out, expected)


def test_select_central_object_keeps_largest_connected_component():
    arr = np.zeros((5, 5), dtype=float)
    arr[1, 1] = 1
    arr[1, 2] = 1
    arr[3, 3] = 1

    out = utils.select_central_object(arr.copy())

    expected = np.zeros((5, 5), dtype=float)
    expected[1, 1] = 1
    expected[1, 2] = 1

    assert np.array_equal(out, expected)


def test_get_central_object_extent_returns_extent_of_central_object():
    arr = np.zeros((5, 5))
    arr[0, 0] = 1
    arr[2, 2] = 3
    arr[2, 3] = 3
    arr[3, 2] = 3
    arr[3, 3] = 3

    extent = utils.get_central_object_extent(arr)
    assert extent == [2, 2]


def test_read_config_parses_values_and_ignores_comments():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = os.path.join(tmpdir, "config.txt")
        with open(cfg, "w") as f:
            f.write("# comment\n")
            f.write("/ another comment\n")
            f.write("a = 1\n")
            f.write('b = "hello"\n')
            f.write("c = (1, 2, 3)\n")
            f.write("d = [4, 5]\n")
            f.write("e = True\n")

        out = utils.read_config(cfg)

        assert out["a"] == 1
        assert out["b"] == "hello"
        assert out["c"] == [1, 2, 3]
        assert out["d"] == [4, 5]
        assert out["e"] is True


def test_read_config_returns_none_for_missing_file():
    out = utils.read_config("definitely_missing_config_file.txt")
    assert out is None


def test_write_config_writes_values_that_read_config_can_parse():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = os.path.join(tmpdir, "config.txt")
        params = {
            "a": 1,
            "b": "hello",
            "c": [1, 2],
            "d": True,
        }

        utils.write_config(params, cfg)
        out = utils.read_config(cfg)

        assert out["a"] == 1
        assert out["b"] == "hello"
        assert out["c"] == [1, 2]
        assert out["d"] is True


def test_save_metrics_writes_metrics_and_errors_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        errs = [0.1, 0.05, 0.01]
        metrics = {"snr": 12.3, "sharpness": 0.9}

        utils.save_metrics(errs, tmpdir, metrics)

        summary = os.path.join(tmpdir, "summary")
        assert os.path.isfile(summary)

        with open(summary, "r") as f:
            text = f.read()

        assert "metric" in text
        assert "snr = 12.3" in text
        assert "sharpness = 0.9" in text
        assert "errors by iteration" in text


def test_write_plot_errors_creates_executable_python_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.write_plot_errors(tmpdir)

        plot_file = os.path.join(tmpdir, "plot_errors.py")
        assert os.path.isfile(plot_file)

        with open(plot_file, "r") as f:
            content = f.read()

        assert "import matplotlib.pyplot as plt" in content
        assert "errs = np.load" in content

        mode = os.stat(plot_file).st_mode
        assert bool(mode & stat.S_IEXEC)


def test_get_logger_creates_default_log_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = utils.get_logger("test_logger_utils", tmpdir)
        logger.debug("hello test log")

        log_file = os.path.join(tmpdir, "default.log")
        assert os.path.isfile(log_file)

        with open(log_file, "r") as f:
            text = f.read()

        assert "hello test log" in text

# Run with pytest
#
# pytest -q
# Notes
# A few caveats about the implementation under test:
#
# crop_center() is not safe for 1D/2D arrays because it loops over range(3) and indexes shape[i].
# select_central_object() selects the largest connected component, not necessarily the most central one.
# save_metrics() appears to write str({er}) literally rather than the numeric error value.
# get_logger() may add duplicate handlers if reused with the same logger name across tests.
# If you want, I can also give you:
#
# a pytest version with tmp_path fixtures,
# mocked tests for get_lib, read_tif, and save_tif,
# or a more complete coverage-oriented suite.
