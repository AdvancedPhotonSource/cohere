import os
import sys
import types
import numpy as np
import pytest

# -----------------------------------------------------------------------------
# Create a fake cohere_core.utilities.utils module before importing alien_tools
# -----------------------------------------------------------------------------
fake_utils = types.ModuleType("cohere_core.utilities.utils")
fake_utils.join = os.path.join


def _pad_center(arr, target_shape):
    result = np.zeros(target_shape, dtype=arr.dtype)
    src_shape = arr.shape
    starts = [(t - s) // 2 for t, s in zip(target_shape, src_shape)]
    slices = tuple(slice(st, st + s) for st, s in zip(starts, src_shape))
    result[slices] = arr
    return result


fake_utils.pad_center = _pad_center

cohere_core = types.ModuleType("cohere_core")
utilities = types.ModuleType("cohere_core.utilities")
utilities.utils = fake_utils
cohere_core.utilities = utilities

sys.modules["cohere_core"] = cohere_core
sys.modules["cohere_core.utilities"] = utilities
sys.modules["cohere_core.utilities.utils"] = fake_utils

# Import module under test after stubbing dependency
import alien_tools


# -----------------------------------------------------------------------------
# Tests for get_asymmetry
# -----------------------------------------------------------------------------
def test_get_asymmetry_symmetric_array():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[1, 1, 1] = 5
    arr[0, 0, 0] = 2
    arr[2, 2, 2] = 2

    asym = alien_tools.get_asymmetry(arr)
    assert np.allclose(asym, 0.0)


def test_get_asymmetry_asymmetric_array():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[0, 0, 0] = 4
    arr[2, 2, 2] = 0

    asym = alien_tools.get_asymmetry(arr)
    assert asym[0, 0, 0] > 0
    assert asym[2, 2, 2] == 0  # only assigned where arr > 0


# -----------------------------------------------------------------------------
# Tests for crop_center
# -----------------------------------------------------------------------------
def test_crop_center_returns_symmetric_crop_around_max():
    arr = np.zeros((7, 5, 3), dtype=float)
    arr[3, 2, 1] = 10  # max in center already

    cropped = alien_tools.crop_center(arr)
    assert cropped.shape == (7, 5, 3)
    assert np.max(cropped) == 10


def test_crop_center_off_center_max():
    arr = np.zeros((7, 7, 7), dtype=float)
    arr[1, 3, 5] = 10

    cropped = alien_tools.crop_center(arr)
    # Expected symmetric crop around center=(1,3,5)
    # half-shapes: min(1,5)=1, min(3,3)=3, min(5,1)=1 -> shape (3,7,3)
    assert cropped.shape == (3, 7, 3)
    assert np.max(cropped) == 10


# -----------------------------------------------------------------------------
# Tests for analyze_clusters
# -----------------------------------------------------------------------------
def test_analyze_clusters_basic():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[0, 0, 0] = 10
    arr[0, 0, 1] = 20
    arr[2, 2, 2] = 5  # noise point

    nz = arr.nonzero()
    labels = np.array([0, 0, -1])  # first two in cluster 0, third is noise

    result = alien_tools.analyze_clusters(arr.copy(), labels, nz)
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
    ) = result

    assert nlabels == 2
    assert labels_arr[0, 0, 0] == 0
    assert labels_arr[2, 2, 2] == -1

    assert noise_arr[2, 2, 2] == 5
    assert no_noise[2, 2, 2] == 0

    assert cluster_size[0, 0, 0] == 2
    assert cluster_size[0, 0, 1] == 2
    assert rel_cluster_size[0, 0, 0] == 1.0
    assert np.isclose(cluster_avg[0, 0, 0], 15.0)
    assert np.isclose(cluster_avg[0, 0, 1], 15.0)

    assert asymmetry.shape == arr.shape
    assert cluster_avg_asym.shape == arr.shape
    assert isinstance(label_counts, tuple)
    assert len(label_counts) == 2


def test_analyze_clusters_does_not_mutate_input():
    arr = np.zeros((3, 3, 3), dtype=float)
    arr[0, 0, 0] = 10
    arr[0, 0, 1] = 20
    arr[2, 2, 2] = 5  # this will be labeled as noise

    original = arr.copy()

    nz = arr.nonzero()
    labels = np.array([0, 0, -1])

    result = alien_tools.analyze_clusters(arr, labels, nz)
    no_noise = result[5]
    noise_arr = result[4]

    assert np.array_equal(arr, original)
    assert noise_arr[2, 2, 2] == 5
    assert no_noise[2, 2, 2] == 0
    assert no_noise[0, 0, 0] == 10
    assert no_noise[0, 0, 1] == 20
# -----------------------------------------------------------------------------
# Tests for save_arr / save_arrays
# -----------------------------------------------------------------------------
def test_save_arr_calls_tifffile_imwrite(monkeypatch, tmp_path):
    called = {}

    def fake_imwrite(path, data):
        called["path"] = path
        called["data"] = data

    monkeypatch.setattr(alien_tools.tif, "imwrite", fake_imwrite)

    arr = np.arange(8).reshape(2, 2, 2).astype(np.float32)
    alien_tools.save_arr(arr, str(tmp_path), "test.tif")

    assert called["path"] == os.path.join(str(tmp_path), "test.tif")
    assert called["data"].shape == arr.transpose().shape
    assert called["data"].dtype == np.float32


def test_save_arrays_calls_save_arr_multiple_times(monkeypatch):
    calls = []

    def fake_save_arr(arr, dir_, fname):
        calls.append((arr, dir_, fname))

    monkeypatch.setattr(alien_tools, "save_arr", fake_save_arr)

    arr = np.zeros((2, 2, 2))
    arrs = (
        2,      # nlabels
        arr,    # 1
        arr,    # 2
        arr,    # 3
        arr,    # 4
        arr,    # 5
        (np.array([0]), np.array([1])),  # 6 label_counts
        arr,    # 7
        arr,    # 8
        arr,    # 9
    )

    alien_tools.save_arrays(arrs, iter=1, thresh=2.5, eps=1.1, dir="out")
    assert len(calls) == 8


# -----------------------------------------------------------------------------
# Tests for remove_blocks
# -----------------------------------------------------------------------------
def test_remove_blocks_zeroes_configured_regions():
    data = np.ones((5, 5, 5), dtype=float)
    config = {
        "aliens": "[(1, 1, 1, 3, 3, 3)]"
    }

    result = alien_tools.remove_blocks(data.copy(), config)

    assert np.all(result[1:3, 1:3, 1:3] == 0)
    assert result[0, 0, 0] == 1


def test_remove_blocks_no_aliens_key():
    data = np.ones((3, 3, 3), dtype=float)
    result = alien_tools.remove_blocks(data.copy(), {})
    assert np.array_equal(result, data)


# -----------------------------------------------------------------------------
# Tests for filter_aliens
# -----------------------------------------------------------------------------
def test_filter_aliens_applies_mask(tmp_path):
    data = np.arange(8).reshape(2, 2, 2).astype(float)
    mask = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]])

    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, mask)

    config = {"alien_file": str(mask_file)}
    result = alien_tools.filter_aliens(data, config)

    expected = np.where(mask == 1, data, 0.0)
    assert np.array_equal(result, expected)


def test_filter_aliens_missing_file(capsys):
    data = np.ones((2, 2, 2), dtype=float)
    config = {"alien_file": "does_not_exist.npy"}

    result = alien_tools.filter_aliens(data.copy(), config)
    captured = capsys.readouterr()

    assert "alien file does not exist" in captured.out
    assert np.array_equal(result, data)


def test_filter_aliens_shape_mismatch(tmp_path, capsys):
    data = np.ones((2, 2, 2), dtype=float)
    mask = np.ones((3, 3, 3), dtype=float)

    mask_file = tmp_path / "bad_mask.npy"
    np.save(mask_file, mask)

    config = {"alien_file": str(mask_file)}
    result = alien_tools.filter_aliens(data.copy(), config)
    captured = capsys.readouterr()

    assert "mask must be of the same shape as data" in captured.out
    assert result is None


def test_filter_aliens_no_config(capsys):
    data = np.ones((2, 2, 2), dtype=float)
    result = alien_tools.filter_aliens(data.copy(), {})
    captured = capsys.readouterr()

    assert "alien_file parameter not configured" in captured.out
    assert np.array_equal(result, data)


# -----------------------------------------------------------------------------
# Tests for remove_aliens dispatcher
# -----------------------------------------------------------------------------
def test_remove_aliens_dispatch_block_aliens(monkeypatch):
    data = np.ones((2, 2, 2), dtype=float)

    def fake_remove_blocks(d, config):
        return np.zeros_like(d)

    monkeypatch.setattr(alien_tools, "remove_blocks", fake_remove_blocks)

    result = alien_tools.remove_aliens(data, {"alien_alg": "block_aliens"})
    assert np.array_equal(result, np.zeros_like(data))


def test_remove_aliens_dispatch_alien_file(monkeypatch):
    data = np.ones((2, 2, 2), dtype=float)

    def fake_filter_aliens(d, config):
        return d * 2

    monkeypatch.setattr(alien_tools, "filter_aliens", fake_filter_aliens)

    result = alien_tools.remove_aliens(data, {"alien_alg": "alien_file"})
    assert np.array_equal(result, data * 2)


def test_remove_aliens_dispatch_autoalien1(monkeypatch):
    data = np.ones((2, 2, 2), dtype=float)

    def fake_auto_alien1(d, config, data_dir=None):
        return d + 1

    monkeypatch.setattr(alien_tools, "auto_alien1", fake_auto_alien1)

    result = alien_tools.remove_aliens(data, {"alien_alg": "AutoAlien1"}, data_dir="x")
    assert np.array_equal(result, data + 1)


def test_remove_aliens_unsupported_algorithm(capsys):
    data = np.ones((2, 2, 2), dtype=float)
    result = alien_tools.remove_aliens(data.copy(), {"alien_alg": "unsupported"})
    captured = capsys.readouterr()

    assert "not supported alien removal algorithm" in captured.out
    assert np.array_equal(result, data)


def test_remove_aliens_no_algorithm(capsys):
    data = np.ones((2, 2, 2), dtype=float)
    result = alien_tools.remove_aliens(data.copy(), {})
    captured = capsys.readouterr()

    assert "alien_alg not configured" in captured.out
    assert np.array_equal(result, data)


# -----------------------------------------------------------------------------
# Tests for auto_alien1 with monkeypatched DBSCAN
# -----------------------------------------------------------------------------
def test_auto_alien1_basic(monkeypatch):
    data = np.zeros((5, 5, 5), dtype=float)
    data[2, 2, 2] = 10
    data[2, 2, 3] = 9
    data[0, 0, 0] = 8

    class FakeDBSCAN:
        def __init__(self, eps, metric, min_samples, n_jobs):
            pass

        def fit_predict(self, X):
            # Label all non-zero points as a single cluster
            return np.zeros(len(X), dtype=int)

    fake_cluster_module = types.ModuleType("sklearn.cluster")
    fake_cluster_module.DBSCAN = FakeDBSCAN
    fake_sklearn_module = types.ModuleType("sklearn")
    fake_sklearn_module.cluster = fake_cluster_module

    monkeypatch.setitem(sys.modules, "sklearn", fake_sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", fake_cluster_module)

    config = {
        "AA1_size_threshold": 0.01,
        "AA1_asym_threshold": 100.0,
        "AA1_min_pts": 1,
        "AA1_eps": 1.1,
        "AA1_amp_threshold": 1,
    }

    result = alien_tools.auto_alien1(data, config, data_dir=".")

    assert result.shape == data.shape
    assert np.count_nonzero(result) > 0


def test_auto_alien1_requires_data_dir():
    data = np.zeros((3, 3, 3), dtype=float)
    config = {}

    with pytest.raises(AttributeError):
        alien_tools.auto_alien1(data, config, data_dir=None)

# Notes
# What this test file covers
# get_asymmetry
# analyze_clusters
# crop_center
# save_arr
# save_arrays
# remove_blocks
# filter_aliens
# remove_aliens
# auto_alien1 (basic behavior with mocked DBSCAN)
# Important implementation detail
# Your module imports:
#
#
# import cohere_core.utilities.utils as ut
# Since that package may not exist in your test environment, the test file injects a fake module into sys.modules before importing alien_tools.
#
# How to run
# From the directory containing both files:
#
#
# pytest -q
# Optional recommendation
# There is a likely bug in analyze_clusters:
#
#
# no_noise = arr
# This does not create a copy; it aliases the original input array. Then:
#
#
# no_noise[noise_pts] = 0
# also modifies arr.
#
# You may want to change it to:
#
#
# no_noise = arr.copy()
# If you make that fix, I can also give you an updated test that explicitly checks that analyze_clusters() does not mutate its input.
#
# If you want, I can also provide:
#
# a smaller minimal pytest version, or
# a more comprehensive production-grade test suite with fixtures and parametri