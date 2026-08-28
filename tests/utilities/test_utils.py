import os
import stat
import numpy as np
import tempfile
import pytest

import cohere_core.utilities.utils as utils


class DummyDevLib:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def next_fast_len(self, dim):
        # Return mapped value if present, otherwise return dim itself
        return self.mapping.get(dim, dim)


def test_get_good_dim_returns_even_fast_len(monkeypatch):
    """
    If next_fast_len already returns an even number >= dim,
    get_good_dim should return it directly.
    """
    dummy_lib = DummyDevLib({5: 6})

    monkeypatch.setattr(utils, "get_lib", lambda pkg: dummy_lib)

    result = utils.get_good_dim(5, "np")

    assert result == 6


def test_get_good_dim_skips_odd_fast_len(monkeypatch):
    """
    If next_fast_len returns an odd number, get_good_dim should keep
    advancing until it finds an even fast length.
    """
    dummy_lib = DummyDevLib({
        5: 5,  # odd
        6: 7,  # odd again
        8: 8,  # finally even
    })

    monkeypatch.setattr(utils, "get_lib", lambda pkg: dummy_lib)

    result = utils.get_good_dim(5, "np")

    assert result == 8


def test_array_to_good_dims_calls_pad_center_with_good_shape(monkeypatch):
    """
    array_to_good_dims should compute good dims for each axis and pass
    the resulting shape into pad_center.
    """
    arr = np.zeros((5, 7))

    monkeypatch.setattr(utils, "get_good_dim", lambda d, pkg: d + 2)

    captured = {}

    def fake_pad_center(input_arr, new_shape):
        captured["arr"] = input_arr
        captured["shape"] = new_shape
        return "padded-array"

    monkeypatch.setattr(utils, "pad_center", fake_pad_center)

    result = utils.array_to_good_dims(arr, "np")

    assert result == "padded-array"
    assert captured["arr"] is arr
    assert captured["shape"] == (7, 9)


def test_adjust_dimensions_positive_pad_1d():
    """
    Positive pads should add zeros around the array.
    1D input is temporarily expanded to 3D and then squeezed back.
    """
    arr = np.array([1, 2, 3])
    pads = [(1, 2)]

    result = utils.adjust_dimensions(arr, pads)

    expected = np.array([0, 1, 2, 3, 0, 0])
    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_negative_pad_1d():
    """
    Negative pads should crop the array.
    """
    arr = np.array([1, 2, 3, 4, 5])
    pads = [(-1, -2)]

    result = utils.adjust_dimensions(arr, pads)

    expected = np.array([2, 3])
    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_mixed_pad_1d():
    """
    Mixed negative and positive pads should crop one side and pad the other.
    """
    arr = np.array([1, 2, 3, 4])
    pads = [(-1, 2)]

    result = utils.adjust_dimensions(arr, pads)

    expected = np.array([2, 3, 4, 0, 0])
    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_positive_pad_2d():
    """
    2D arrays should be expanded internally and then squeezed back to 2D.
    """
    arr = np.array([[1, 2], [3, 4]])
    pads = [(1, 1), (2, 0)]

    result = utils.adjust_dimensions(arr, pads)

    expected = np.array([
        [0, 0, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4],
        [0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_crop_2d():
    """
    Cropping in 2D should remove rows/columns as specified.
    """
    arr = np.arange(1, 17).reshape(4, 4)
    pads = [(-1, -1), (-1, -1)]

    result = utils.adjust_dimensions(arr, pads)

    expected = np.array([
        [6, 7],
        [10, 11],
    ])
    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_3d_mixed():
    """
    Test mixed padding/cropping on a true 3D array.
    """
    arr = np.arange(27).reshape(3, 3, 3)
    pads = [(1, 0), (-1, 1), (0, 2)]

    result = utils.adjust_dimensions(arr, pads)

    # Manually build expected result
    cropped = arr[:, 1:, :]  # crop second axis from front by 1
    expected = np.pad(
        cropped,
        pad_width=[(1, 0), (0, 1), (0, 2)],
        mode="constant",
        constant_values=0,
    )

    np.testing.assert_array_equal(result, expected)


def test_adjust_dimensions_no_change():
    """
    Zero pads should leave the array unchanged.
    """
    arr = np.array([[1, 2], [3, 4]])
    pads = [(0, 0), (0, 0)]

    result = utils.adjust_dimensions(arr, pads)

    np.testing.assert_array_equal(result, arr)


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


def test_crop_center_crops_2d_array_around_center():
    arr = np.arange(5 * 5 ).reshape((5, 5))
    cropped = utils.crop_center(arr, (3, 3))

    expected = arr[1:4, 1:4]
    assert np.array_equal(cropped, expected)


def test_crop_center_crops_1d_array_around_center():
    arr = np.arange(5).reshape((5,))
    cropped = utils.crop_center(arr, (3,))

    expected = arr[1:4]
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
            f.write("f = {'a': 1, 'b': 2}\n")

        out = utils.read_config(cfg)

        assert out["a"] == 1
        assert out["b"] == "hello"
        assert out["c"] == [1, 2, 3]
        assert out["d"] == [4, 5]
        assert out["e"] is True
        assert out["f"] == {'a': 1, 'b': 2}


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
