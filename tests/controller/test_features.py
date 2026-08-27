import numpy as np
import pytest

import features


class DummyDevLib:
    @staticmethod
    def load(path):
        return np.load(path)

    @staticmethod
    def dims(data):
        return data.shape

    @staticmethod
    def fftshift(data):
        return np.fft.fftshift(data)

    @staticmethod
    def sum(data):
        return np.sum(data)

    @staticmethod
    def square(data):
        return np.square(data)

    @staticmethod
    def full(shape, value, dtype=None):
        return np.full(shape, value, dtype=dtype)

    @staticmethod
    def dtype(data):
        return data.dtype

    @staticmethod
    def sqrt(data):
        return np.sqrt(data)

    @staticmethod
    def fftconvolve(a, b):
        # lightweight stub for testing
        return a + b

    @staticmethod
    def gaussian_filter(data, sigma):
        # deterministic stub for testing
        return data + sigma

    @staticmethod
    def angle(data):
        return np.angle(data)


@pytest.fixture(autouse=True)
def patch_devlib():
    features.set_lib(DummyDevLib)


@pytest.fixture
def patch_utils(monkeypatch):
    def crop_center(arr, shape):
        slices = tuple(slice(0, s) for s in shape)
        return arr[slices]

    monkeypatch.setattr(features.ut, "crop_center", crop_center)
    monkeypatch.setattr(features.ut, "join", lambda a, b: f"{a}/{b}")


@pytest.fixture
def patch_dvut(monkeypatch):
    monkeypatch.setattr(
        features.dvut,
        "lucy_deconvolution",
        lambda amplitudes, roi_data, kernel, iterations: kernel + 1
    )
    monkeypatch.setattr(
        features.dvut,
        "shrink_wrap",
        lambda ds_image, threshold, gauss_sigma: {
            "image": ds_image,
            "threshold": threshold,
            "sigma": gauss_sigma,
        }
    )


def test_pcdi_init_requires_iterations(patch_utils):
    params = {
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4))

    with pytest.raises(ValueError, match="pc_LUCY_iterations parameter not defined"):
        features.Pcdi(params, data)


def test_pcdi_init_requires_kernel(patch_utils):
    params = {
        "pc_LUCY_iterations": 5,
    }
    data = np.ones((4, 4))

    with pytest.raises(ValueError, match="pc_LUCY_kernel parameter not defined"):
        features.Pcdi(params, data)


def test_pcdi_init_sets_default_kernel_when_no_dir(patch_utils):
    params = {
        "pc_LUCY_iterations": 5,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4), dtype=np.float32)

    obj = features.Pcdi(params, data)

    assert obj.kernel.shape == (2, 2)
    assert np.all(obj.kernel == 0.5)
    assert obj.iterations == 5
    assert obj.normalize is True


def test_pcdi_set_previous(patch_utils):
    params = {
        "pc_LUCY_iterations": 5,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4))
    obj = features.Pcdi(params, data)

    abs_amplitudes = np.arange(16).reshape(4, 4)
    obj.set_previous(abs_amplitudes)

    assert obj.roi_amplitudes_prev.shape == (2, 2)


def test_pcdi_apply_partial_coherence(patch_utils):
    params = {
        "pc_LUCY_iterations": 5,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4))
    obj = features.Pcdi(params, data)

    abs_amplitudes = np.ones((2, 2))
    result = obj.apply_partial_coherence(abs_amplitudes)

    expected = np.sqrt(np.square(abs_amplitudes) + obj.kernel)
    np.testing.assert_allclose(result, expected)


def test_pcdi_update_partial_coherence_calls_lucy(patch_utils, patch_dvut):
    params = {
        "pc_LUCY_iterations": 3,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4))
    obj = features.Pcdi(params, data)
    obj.roi_amplitudes_prev = np.ones((2, 2))

    old_kernel = obj.kernel.copy()
    obj.update_partial_coherence(np.ones((4, 4)) * 2)

    np.testing.assert_array_equal(obj.kernel, old_kernel + 1)


def test_lowpass_filter_builds_sigmas():
    params = {
        "lowpass_filter_trigger": [0, 1, 4],
        "lowpass_filter_range": [0.5, 1.5],
    }
    obj = features.LowPassFilter(params)

    assert obj.filter_sigmas == [0.5, 0.75, 1.0, 1.25]


def test_lowpass_filter_apply_trigger():
    params = {
        "lowpass_filter_trigger": [0, 1, 3],
        "lowpass_filter_range": [1.0, 3.0],
    }
    obj = features.LowPassFilter(params)
    data = np.ones((2, 2))

    result = obj.apply_trigger(data, 1)
    np.testing.assert_allclose(result, data + obj.filter_sigmas[1])


def test_triggeredop_general_trigger_shrinkwrap(patch_dvut):
    params = {
        "shrink_wrap_trigger": [0, 1, 5],
        "shrink_wrap_gauss_sigma": 2.0,
        "shrink_wrap_threshold": 0.1,
    }
    trig_info = {}

    obj = features.create("shrink_wrap", params, trig_info)
    result = obj.apply_trigger(np.ones((2, 2)))

    assert result["threshold"] == 0.1
    assert result["sigma"] == 2.0


def test_triggeredop_subtrigger_sequence():
    class DummyRow:
        def tolist(self):
            return [1, 2, 1, 0]

    class DummyTriggered(features.TriggeredOp):
        class Op:
            def __init__(self, value):
                self.value = value

            def apply_trigger(self, x):
                return x + self.value

        def create_obj(self, params, index=None, beg=None, end=None):
            return self.Op(index if index is not None else params["default"])

    obj = DummyTriggered("dummy")
    params = {"default": 100}
    trig_info = {
        "dummy_trigger": (
            DummyRow(),
            [(0, 2, 1), (2, 4, 2)],
        )
    }

    obj.create_objs(params, trig_info)

    assert obj.apply_trigger(10) == 11
    assert obj.apply_trigger(10) == 12
    assert obj.apply_trigger(10) == 11


def test_shrinkwrap_create_obj_requires_sigma():
    obj = features.ShrinkWrapGauss("shrink_wrap")
    params = {
        "shrink_wrap_threshold": 0.1
    }

    with pytest.raises(ValueError, match="shrink_wrap_gauss_sigma parameter not defined"):
        obj.create_obj(params)


def test_shrinkwrap_create_obj_requires_threshold():
    obj = features.ShrinkWrapGauss("shrink_wrap")
    params = {
        "shrink_wrap_gauss_sigma": 2.0
    }

    with pytest.raises(ValueError, match="shrink_wrap_threshold parameter not defined"):
        obj.create_obj(params)


def test_phaseconstrain_requires_phase_min():
    obj = features.PhaseConstrain("phc")
    params = {
        "phc_phase_max": 1.0
    }

    with pytest.raises(ValueError, match="phc_phase_min parameter not defined"):
        obj.create_obj(params)


def test_phaseconstrain_requires_phase_max():
    obj = features.PhaseConstrain("phc")
    params = {
        "phc_phase_min": -1.0
    }

    with pytest.raises(ValueError, match="phc_phase_max parameter not defined"):
        obj.create_obj(params)


def test_phaseconstrain_apply_trigger():
    obj = features.PhaseConstrain("phc")
    params = {
        "phc_phase_min": -0.5,
        "phc_phase_max": 0.5,
    }
    phc = obj.create_obj(params)

    data = np.array([1 + 0j, 1j, -1 + 0j])
    result = phc.apply_trigger(data)

    expected = (np.angle(data) > -0.5) & (np.angle(data) < 0.5)
    np.testing.assert_array_equal(result, expected)


def test_globalmin_tracks_best_image():
    gm = features.GlobalMin("global_min")
    best = gm.create_obj({})

    img1 = np.array([[1]])
    img2 = np.array([[2]])

    best.apply_trigger(img1, 0.4)
    best.apply_trigger(img2, 0.2)

    best_image, min_error = gm.get_best()
    np.testing.assert_array_equal(best_image, img2)
    assert min_error == 0.2


def test_create_returns_shrinkwrap(patch_dvut):
    params = {
        "shrink_wrap_trigger": [0, 1, 5],
        "shrink_wrap_gauss_sigma": 2.0,
        "shrink_wrap_threshold": 0.1,
    }

    obj = features.create("shrink_wrap", params, {})
    assert isinstance(obj, features.ShrinkWrapGauss)


def test_create_returns_phaseconstrain():
    params = {
        "phc_trigger": [0, 1, 5],
        "phc_phase_min": -1.0,
        "phc_phase_max": 1.0,
    }

    obj = features.create("phc", params, {})
    assert isinstance(obj, features.PhaseConstrain)


def test_create_returns_globalmin():
    params = {
        "global_min_trigger": [0, 1, 5],
    }

    obj = features.create("global_min", params, {})
    assert isinstance(obj, features.GlobalMin)


# Notes
# A few important points about this test file:
#
# It assumes your module is named features.py and importable as:
#
#
# import features
# Since your code depends on:
#
# cohere_core.utilities.utils as ut
# cohere_core.utilities.dvc_utils as dvut
# a global devlib
# the tests mock/stub those pieces so the tests can run in isolation.
#
# DummyDevLib.fftconvolve() and gaussian_filter() are simplified deterministic stubs, which is usually best for unit tests.
#
# How to run
# From the directory containing features.py and test_features.py:
#
#
# pytest -q
# Optional improvement
# There is one issue in create():
#
#
# if trig_op == 'shrink_wrap':
#     to = ShrinkWrapGauss(trig_op)
# if trig_op == 'phc':
#     to = PhaseConstrain(trig_op)
# if trig_op == 'global_min':
#     to = GlobalMin(trig_op)
# This should ideally be if/elif/elif, and it should probably raise an error for unknown trig_op, e.g.:
#
#
def create(trig_op, params, trig_op_info):
    if trig_op == 'shrink_wrap':
        to = ShrinkWrapGauss(trig_op)
    elif trig_op == 'phc':
        to = PhaseConstrain(trig_op)
    elif trig_op == 'global_min':
        to = GlobalMin(trig_op)
    else:
        raise ValueError(f"Unsupported trigger op: {trig_op}")

    to.create_objs(params, trig_op_info)
    return to
# If you want, I can also rewrite these tests to:
#
# use unittest.mock instead of fixtures/stubs, or
# be more minimal, or
# target 100% coverage for this file.