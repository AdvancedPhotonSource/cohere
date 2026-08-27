import importlib
import types

import numpy as np
import pytest


@pytest.fixture
def dvc_utils(monkeypatch):
    """
    Import dvc_utils and replace its devlib with a lightweight numpy-backed stub.
    """
    module = importlib.import_module("cohere_core.utilities.dvc_utils")

    class DummyDevLib:
        def __init__(self):
            self.random = lambda shape: np.random.random(shape)

        def to_numpy(self, x):
            return np.asarray(x)

        def from_numpy(self, x, device=None):
            return np.asarray(x)

        def correlate(self, a, b, mode="same", method="fft"):
            # simple FFT-based correlation for same-shaped arrays
            fa = np.fft.fftn(a)
            fb = np.fft.fftn(b)
            cc = np.fft.ifftn(fa * np.conj(fb))
            return cc

        def ifftshift(self, x):
            return np.fft.ifftshift(x)

        def fftshift(self, x):
            return np.fft.fftshift(x)

        def fft(self, x):
            return np.fft.fftn(x)

        def ifft(self, x):
            return np.fft.ifftn(x)

        def dims(self, x):
            return x.shape

        def absolute(self, x):
            return np.abs(x)

        def unravel_index(self, idx, shape):
            return np.unravel_index(idx, shape)

        def argmax(self, x):
            return np.argmax(x)

        def sum(self, x, axis=None):
            return np.sum(x, axis=axis)

        def where(self, cond, x, y):
            return np.where(cond, x, y)

        def full(self, shape, fill_value, device=None):
            return np.full(shape, fill_value)

        def square(self, x):
            return np.square(x)

        def array_equal(self, a, b):
            return np.array_equal(a, b)

        def angle(self, x):
            return np.angle(x)

        def sqrt(self, x):
            return np.sqrt(x)

        def exp(self, x):
            return np.exp(x)

        def maximum(self, a, b):
            return np.maximum(a, b)

        def amax(self, x):
            return np.amax(x)

        def amin(self, x):
            return np.amin(x)

        def histogram2d(self, x, y, bins):
            h, _, _ = np.histogram2d(x, y, bins=bins)
            return h

        def ravel(self, x):
            return np.ravel(x)

        def geomspace(self, start, stop, num):
            return np.geomspace(start, stop, num)

        def reshape(self, x, shape):
            return np.reshape(x, shape)

        def entropy(self, x):
            x = np.asarray(x, dtype=float)
            total = x.sum()
            if total == 0:
                return 0.0
            p = x / total
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        def meshgrid(self, *args, **kwargs):
            return np.meshgrid(*args, **kwargs)

        def abs(self, x):
            return np.abs(x)

        def roll(self, x, shift, axis=None):
            shift = tuple(int(s) for s in shift) if isinstance(shift, (list, tuple)) else int(shift)
            return np.roll(x, shift, axis=axis)

        def center_of_mass(self, x):
            x = np.asarray(x, dtype=float)
            total = x.sum()
            inds = np.indices(x.shape)
            return tuple((inds[i] * x).sum() / total for i in range(x.ndim))

        def conj(self, x):
            return np.conj(x)

        def gaussian_filter(self, arr, sigma):
            # lightweight fallback: identity for tests that do not care about blur details
            return np.asarray(arr)

        def pad(self, arr, pad_width):
            return np.pad(arr, pad_width)

        def array(self, x, device=None):
            return np.array(x)

        def real(self, x):
            return np.real(x)

        def clip(self, x, a_min, a_max=None):
            return np.clip(x, a_min, a_max)

        def flip(self, x):
            return np.flip(x)

    monkeypatch.setattr(module, "devlib", DummyDevLib(), raising=False)
    return module


def test_use_numpy_decorator_converts_array_like_args(dvc_utils, monkeypatch):
    calls = {}

    @dvc_utils.use_numpy
    def func(a, b):
        calls["a_type"] = type(a)
        calls["b_type"] = type(b)
        return a.shape, b

    arr = np.array([[1, 2], [3, 4]])
    shape, b = func(arr, 5)

    assert shape == (2, 2)
    assert isinstance(b, int)
    assert calls["a_type"] is np.ndarray


def test_fast_shift_positive_shift(dvc_utils):
    arr = np.array([[1, 2], [3, 4]])
    shifted = dvc_utils.fast_shift(arr, [1, 0], fill_val=0)

    expected = np.array([[0, 0],
                         [1, 2]])
    np.testing.assert_array_equal(shifted, expected)


def test_fast_shift_negative_shift(dvc_utils):
    arr = np.array([[1, 2], [3, 4]])
    shifted = dvc_utils.fast_shift(arr, [-1, 0], fill_val=0)

    expected = np.array([[3, 4],
                         [0, 0]])
    np.testing.assert_array_equal(shifted, expected)


def test_pad_around_centers_array(dvc_utils):
    arr = np.array([[1, 2], [3, 4]])
    padded = dvc_utils.pad_around(arr, (4, 4), val=0)

    expected = np.array([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
    ])
    np.testing.assert_array_equal(padded, expected)


def test_pad_around_raises_on_smaller_shape(dvc_utils):
    arr = np.ones((3, 3))
    with pytest.raises(ValueError, match="greater than or equal"):
        dvc_utils.pad_around(arr, (2, 3))


def test_get_norm_returns_sum_of_squared_magnitudes(dvc_utils):
    arr = np.array([1 + 1j, 2 + 0j])
    result = dvc_utils.get_norm(arr)

    expected = np.abs(1 + 1j) ** 2 + np.abs(2 + 0j) ** 2
    assert result == pytest.approx(expected)


def test_histogram2d_returns_expected_shape(dvc_utils):
    arr1 = np.array([[1, 2], [3, 4]], dtype=float)
    arr2 = np.array([[1, 2], [3, 4]], dtype=float)

    hist = dvc_utils.histogram2d(arr1, arr2, n_bins=5, log=False)

    assert hist.shape == (5, 5)
    assert hist.sum() == arr1.size


def test_calc_ehd_zero_for_identical_histogram_mass_on_diagonal(dvc_utils):
    hgram = np.eye(4)
    ehd = dvc_utils.calc_ehd(hgram)

    assert ehd == pytest.approx(0.0)


def test_correlation_err_zero_for_identical_arrays(dvc_utils):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    err = dvc_utils.correlation_err(arr, arr)

    assert err == pytest.approx(0.0)


def test_align_arrays_pixel_returns_same_array_when_shift_is_zero(dvc_utils):
    arr = np.array([[1, 2], [3, 4]])
    shift = np.array([0, 0])

    result = dvc_utils.align_arrays_pixel(arr, arr, shift=shift)

    np.testing.assert_array_equal(result, arr)


def test_get_metric_chi(dvc_utils):
    image = np.array([[1.0]])
    errs = [0.5, 0.25]

    result = dvc_utils.get_metric(image, errs, "chi")

    assert result == pytest.approx(0.25)


def test_get_metric_sharpness(dvc_utils):
    image = np.array([[2.0]])
    errs = [0.5]

    result = dvc_utils.get_metric(image, errs, "sharpness")

    # sum((abs(image)^2)^2) = (4)^2 = 16
    assert result == pytest.approx(16.0)


def test_pad_to_cube_pads_small_array(dvc_utils):
    arr = np.ones((2, 2, 2))
    result = dvc_utils.pad_to_cube(arr, 4)

    assert result.shape == (4, 4, 4)
    assert result[1:3, 1:3, 1:3].sum() == 8


def test_shift_phase_preserves_magnitude(dvc_utils, monkeypatch):
    monkeypatch.setattr(dvc_utils, "shrink_wrap", lambda arr, t, s: np.ones_like(arr))

    arr = np.array([1 + 1j, 1 - 1j])
    shifted = dvc_utils.shift_phase(arr, val=0)

    np.testing.assert_allclose(np.abs(shifted), np.abs(arr))


def test_zero_phase_preserves_magnitude(dvc_utils, monkeypatch):
    monkeypatch.setattr(dvc_utils, "shift_phase", lambda arr, val=0: np.array([1j, 1j]))

    arr = np.array([3 + 4j, 1 + 0j])
    result = dvc_utils.zero_phase(arr)

    np.testing.assert_allclose(np.abs(result), np.abs(arr))


def test_zero_phase_cc_aligns_global_phase(dvc_utils):
    arr1 = np.array([1 + 0j, 1 + 0j])
    arr2 = np.array([1j, 1j])

    result = dvc_utils.zero_phase_cc(arr1, arr2)

    np.testing.assert_allclose(result, np.array([1j, 1j]))

# Notes
# A few things to be aware of with this module:
#
# devlib is global
#
# Tests must inject or monkeypatch it.
# The fixture above handles that.
# Some functions are harder to unit test cleanly because they depend on:
#
# FFT conventions
# device-specific array APIs
# scipy-like image ops hidden behind devlib
# There is a duplicate definition of calc_nmi in your file.
#
# The second definition overrides the first.
# Your tests should assume the second one is the active function.
# How to run
#
# pytest -q
# Optional improvement
# If you want, I can also provide:
#
# a more minimal test file
# a more comprehensive test suite
# tests rewritten to use a mocked ut.get_lib('np') path
# a conftest.py version that shares the DummyDevLib fixture across tests
# If you want, I can generate a version tailored to your exact package layout.