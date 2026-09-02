# It includes tests for:
#
# get_asymmetry
# analyze_clusters
# crop_center
# remove_blocks
# filter_aliens
# remove_aliens
# auto_alien1 with mocking so it stays lightweight
# You can save this as something like:
#
#
# tests/test_alien_tools.py
#
import numpy as np
import pytest

from cohere_core.data import alien_tools


def test_get_asymmetry_symmetric_array():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[1, 1, 1] = 5
    arr[0, 0, 0] = 2
    arr[2, 2, 2] = 2

    asym = alien_tools.get_asymmetry(arr)

    assert asym.shape == arr.shape
    assert np.allclose(asym, 0.0)


def test_get_asymmetry_asymmetric_array():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[0, 0, 0] = 4
    arr[2, 2, 2] = 2

    asym = alien_tools.get_asymmetry(arr)

    expected = abs(4 - 2) / ((4 + 2) / 2.0)
    assert np.isclose(asym[0, 0, 0], expected)
    # output is only assigned where arr > 0
    assert np.isclose(asym[2, 2, 2], expected)
    assert np.all(asym[arr == 0] == 0)


def test_analyze_clusters_basic():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[0, 0, 0] = 10
    arr[0, 0, 1] = 20
    arr[2, 2, 2] = 5

    nz = arr.nonzero()
    labels = np.array([0, 0, -1])

    (
        nlabels,
        labels_arr,
        rel_cluster_size,
        cluster_avg,
        noise_arr,
        no_noise,
        label_counts,
        cluster_avg_asym,
        asymmetry,
        cluster_size,
    ) = alien_tools.analyze_clusters(arr, labels, nz)

    assert nlabels == 2
    assert labels_arr[0, 0, 0] == 0
    assert labels_arr[0, 0, 1] == 0
    assert labels_arr[2, 2, 2] == -1

    assert noise_arr[2, 2, 2] == 5
    assert no_noise[2, 2, 2] == 0

    assert cluster_size[0, 0, 0] == 2
    assert cluster_size[0, 0, 1] == 2
    assert rel_cluster_size[0, 0, 0] == 1
    assert rel_cluster_size[0, 0, 1] == 1

    assert np.isclose(cluster_avg[0, 0, 0], 15.0)
    assert np.isclose(cluster_avg[0, 0, 1], 15.0)

    assert asymmetry.shape == arr.shape
    assert cluster_avg_asym.shape == arr.shape
    assert label_counts[0].tolist() == [-1, 0]
    assert label_counts[1].tolist() == [1, 2]


def test_crop_center_returns_symmetric_subarray_around_max():
    arr = np.zeros((7, 5, 6), dtype=float)
    arr[3, 2, 4] = 10  # max element

    cropped = alien_tools.crop_center(arr)

    # max should be centered within the cropped array
    center = np.unravel_index(np.argmax(cropped), cropped.shape)
    expected_center = tuple(s // 2 for s in cropped.shape)
    assert center == expected_center

    # result must be odd-sized in all dimensions to keep max centered
    assert all(s % 2 == 1 for s in cropped.shape)


def test_remove_blocks_zeroes_specified_regions():
    data = np.ones((5, 5, 5), dtype=float)
    config = {
        "aliens": [
            [1, 1, 1, 3, 3, 3],
        ]
    }

    result = alien_tools.remove_blocks(data.copy(), config)

    assert np.all(result[1:3, 1:3, 1:3] == 0)
    assert result[0, 0, 0] == 1
    assert result[4, 4, 4] == 1


def test_remove_blocks_no_aliens_key_returns_unchanged():
    data = np.ones((4, 4, 4), dtype=float)
    result = alien_tools.remove_blocks(data.copy(), {})
    assert np.array_equal(result, data)


def test_filter_aliens_applies_mask(tmp_path):
    data = np.arange(27, dtype=float).reshape((3, 3, 3))
    mask = np.zeros((3, 3, 3), dtype=int)
    mask[1, 1, 1] = 1
    mask[2, 2, 2] = 1

    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)

    config = {"alien_file": str(mask_file)}
    result = alien_tools.filter_aliens(data.copy(), config)

    expected = np.zeros_like(data)
    expected[1, 1, 1] = data[1, 1, 1]
    expected[2, 2, 2] = data[2, 2, 2]

    assert np.array_equal(result, expected)


def test_filter_aliens_missing_file_returns_original_and_prints(capsys):
    data = np.ones((3, 3, 3), dtype=float)
    config = {"alien_file": "does_not_exist.npy"}

    result = alien_tools.filter_aliens(data.copy(), config)

    captured = capsys.readouterr()
    assert "alien file does not exist" in captured.out
    assert np.array_equal(result, data)


def test_filter_aliens_shape_mismatch_returns_none(tmp_path, capsys):
    data = np.ones((3, 3, 3), dtype=float)
    mask = np.ones((2, 2, 2), dtype=int)

    mask_file = tmp_path / "bad_mask.npy"
    np.save(mask_file, mask)

    config = {"alien_file": str(mask_file)}
    result = alien_tools.filter_aliens(data.copy(), config)

    captured = capsys.readouterr()
    assert "mask must be of the same shape as data" in captured.out
    assert result is None


def test_remove_aliens_dispatch_block_aliens():
    data = np.ones((4, 4, 4), dtype=float)
    config = {
        "alien_alg": "block_aliens",
        "aliens": [[1, 1, 1, 3, 3, 3]],
    }

    result = alien_tools.remove_aliens(data.copy(), config)

    assert np.all(result[1:3, 1:3, 1:3] == 0)
    assert result[0, 0, 0] == 1


def test_remove_aliens_dispatch_alien_file(tmp_path):
    data = np.arange(8, dtype=float).reshape((2, 2, 2))
    mask = np.zeros((2, 2, 2), dtype=int)
    mask[0, 0, 0] = 1

    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)

    config = {
        "alien_alg": "alien_file",
        "alien_file": str(mask_file),
    }

    result = alien_tools.remove_aliens(data.copy(), config)

    expected = np.zeros_like(data)
    expected[0, 0, 0] = data[0, 0, 0]
    assert np.array_equal(result, expected)


def test_remove_aliens_dispatch_autoalien1(monkeypatch):
    data = np.ones((3, 3, 3), dtype=float)
    config = {"alien_alg": "AutoAlien1"}

    expected = np.full_like(data, 7.0)

    def fake_auto_alien1(data_arg, config_arg, data_dir_arg):
        assert np.array_equal(data_arg, data)
        assert config_arg == config
        return expected

    monkeypatch.setattr(alien_tools, "auto_alien1", fake_auto_alien1)

    result = alien_tools.remove_aliens(data.copy(), config, data_dir="some_dir")

    assert np.array_equal(result, expected)


def test_remove_aliens_unsupported_algorithm_prints_and_returns_original(capsys):
    data = np.ones((3, 3, 3), dtype=float)
    config = {"alien_alg": "unsupported"}

    result = alien_tools.remove_aliens(data.copy(), config)

    captured = capsys.readouterr()
    assert "not supported alien removal algorithm" in captured.out
    assert np.array_equal(result, data)


def test_remove_aliens_missing_algorithm_prints_and_returns_original(capsys):
    data = np.ones((3, 3, 3), dtype=float)

    result = alien_tools.remove_aliens(data.copy(), {})

    captured = capsys.readouterr()
    assert "alien_alg not configured" in captured.out
    assert np.array_equal(result, data)


def test_auto_alien1_smoke(monkeypatch):
    data = np.zeros((5, 5, 5), dtype=float)
    data[2, 2, 2] = 10
    config = {
        "AA1_size_threshold": 0.5,
        "AA1_asym_threshold": 1.0,
        "AA1_min_pts": 1,
        "AA1_eps": 1.1,
        "AA1_amp_threshold": 1,
        "AA1_expandcleanedsigma": 0.0,
    }

    class FakeDBSCAN:
        def __init__(self, eps, metric, min_samples, n_jobs):
            self.eps = eps
            self.metric = metric
            self.min_samples = min_samples
            self.n_jobs = n_jobs

        def fit_predict(self, x):
            # single point -> one cluster labeled 0
            return np.zeros(len(x), dtype=int)

    monkeypatch.setitem(__import__("sys").modules, "sklearn.cluster", type("M", (), {"DBSCAN": FakeDBSCAN}))

    # patch pad_center to avoid dependence on external utility behavior
    monkeypatch.setattr(alien_tools.ut, "pad_center", lambda arr, shape: arr)

    result = alien_tools.auto_alien1(data, config, data_dir="test_dir")

    assert result.shape == data.shape
    assert result[2, 2, 2] == 10

# Notes
# 1. About the auto_alien1 test
# Because auto_alien1 imports DBSCAN inside the function:
#
#
# from sklearn.cluster import DBSCAN
# it is a bit trickier to mock. The smoke test above uses a monkeypatch on sys.modules for sklearn.cluster before the function call. That works in many cases, but if sklearn is installed and already imported, a cleaner approach is to refactor the source slightly to import DBSCAN at module level. Then mocking becomes easier.
#
# 2. Potential bug in auto_alien1
# This line:
#
#
# data_dir = data_dir.replace(os.sep, '/')
# will raise an exception if data_dir is None. Since the function signature allows data_dir=None, you may want to change it to:
#
#
# if data_dir is not None:
#     data_dir = data_dir.replace(os.sep, '/')
# If not fixed, tests calling auto_alien1(..., data_dir=None) will fail.
#
# 3. Potential behavior in analyze_clusters
# The loop:
#
#
# for n in range(1, nlabels):
# assumes the first unique label is noise (-1) and skips it. That works when noise exists, but if there is no -1 label, cluster 0 will be skipped. You may want a test for that if this is important, or consider fixing the implementation.
#
# For example, a safer implementation would iterate actual labels and skip only -1.
#
# If you want, I can also provide:
#
# a more minimal test file
# a more thorough test suite with parametrization
# a version adapted for unittest instead of pytest
# tests that target edge cases / likely bugs in this implementation