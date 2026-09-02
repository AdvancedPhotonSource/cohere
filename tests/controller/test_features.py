import numpy as np
import pytest

import cohere_core.controller.features as features


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
    def full(shape, fill_value, dtype=None):
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def dtype(data):
        return data.dtype

    @staticmethod
    def sqrt(data):
        return np.sqrt(data)

    @staticmethod
    def angle(data):
        return np.angle(data)

    @staticmethod
    def gaussian_filter(data, sigma):
        # simple deterministic stand-in for testing
        return data + sigma

    @staticmethod
    def fftconvolve(a, b):
        # simple stand-in preserving shape for test purposes
        return a + np.mean(b)


@pytest.fixture(autouse=True)
def setup_devlib():
    features.set_lib(DummyDevLib)


@pytest.fixture
def patch_utils(monkeypatch):
    def fake_crop_center(data, kernel_area):
        slices = tuple(slice(0, k) for k in kernel_area)
        return data[slices]

    def fake_join(*args):
        return "/".join(args)

    monkeypatch.setattr(features.ut, "crop_center", fake_crop_center)
    monkeypatch.setattr(features.ut, "join", fake_join)


@pytest.fixture
def patch_dvut(monkeypatch):
    calls = {}

    def fake_shrink_wrap(ds_image, threshold, gauss_sigma):
        calls["shrink_wrap"] = (ds_image, threshold, gauss_sigma)
        return np.ones_like(ds_image) * threshold

    def fake_lucy_deconvolution(amplitudes2, roi_data2, kernel, iterations):
        calls["lucy"] = (amplitudes2, roi_data2, kernel, iterations)
        return kernel + 1

    monkeypatch.setattr(features.dvut, "shrink_wrap", fake_shrink_wrap)
    monkeypatch.setattr(features.dvut, "lucy_deconvolution", fake_lucy_deconvolution)
    return calls


def test_create_unsupported_trigger_op():
    with pytest.raises(ValueError, match="Unsupported trigger op"):
        features.create("bad_op", {}, {})


def test_create_shrink_wrap_general_trigger():
    params = {
        "shrink_wrap_trigger": [0, 1, 5],
        "shrink_wrap_gauss_sigma": 2.5,
        "shrink_wrap_threshold": 0.1,
    }

    obj = features.create("shrink_wrap", params, {})
    assert isinstance(obj, features.ShrinkWrapGauss)
    assert obj.f == obj.apply_trigger_obj
    assert isinstance(obj.objs, features.ShrinkWrapGauss.GaussSW)
    assert obj.objs.gauss_sigma == 2.5
    assert obj.objs.threshold == 0.1


def test_shrink_wrap_apply_trigger_calls_dvut(patch_dvut):
    sw = features.ShrinkWrapGauss("shrink_wrap")
    params = {
        "shrink_wrap_trigger": [0, 1, 5],
        "shrink_wrap_gauss_sigma": 3.0,
        "shrink_wrap_threshold": 0.25,
    }
    sw.create_objs(params, {})
    data = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = sw.apply_trigger(data)

    assert np.all(result == 0.25)
    assert patch_dvut["shrink_wrap"][1] == 0.25
    assert patch_dvut["shrink_wrap"][2] == 3.0


def test_shrink_wrap_missing_sigma_raises():
    sw = features.ShrinkWrapGauss("shrink_wrap")
    params = {
        "shrink_wrap_threshold": 0.25,
    }
    with pytest.raises(ValueError, match="shrink_wrap_gauss_sigma parameter not defined"):
        sw.create_obj(params)


def test_shrink_wrap_missing_threshold_raises():
    sw = features.ShrinkWrapGauss("shrink_wrap")
    params = {
        "shrink_wrap_gauss_sigma": 2.0,
    }
    with pytest.raises(ValueError, match="shrink_wrap_threshold parameter not defined"):
        sw.create_obj(params)


def test_phase_constrain_general_trigger():
    params = {
        "phc_trigger": [0, 1, 5],
        "phc_phase_min": -0.5,
        "phc_phase_max": 0.5,
    }

    obj = features.create("phc", params, {})
    assert isinstance(obj, features.PhaseConstrain)
    assert obj.f == obj.apply_trigger_obj

    data = np.array([1 + 0j, 1j, -1 + 0j])
    result = obj.apply_trigger(data)

    expected = np.array([True, False, False])
    np.testing.assert_array_equal(result, expected)


def test_phase_constrain_missing_min_raises():
    phc = features.PhaseConstrain("phc")
    params = {"phc_phase_max": 1.0}
    with pytest.raises(ValueError, match="phc_phase_min parameter not defined"):
        phc.create_obj(params)


def test_phase_constrain_missing_max_raises():
    phc = features.PhaseConstrain("phc")
    params = {"phc_phase_min": -1.0}
    with pytest.raises(ValueError, match="phc_phase_max parameter not defined"):
        phc.create_obj(params)


def test_global_min_tracks_best_image():
    gm = features.GlobalMin("global_min")
    params = {"global_min_trigger": [0, 1, 5]}
    gm.create_objs(params, {})

    img1 = np.array([[1]])
    img2 = np.array([[2]])
    img3 = np.array([[3]])

    gm.apply_trigger(img1, 0.5)
    gm.apply_trigger(img2, 0.2)
    gm.apply_trigger(img3, 0.3)

    best_image, min_error = gm.get_best()
    assert min_error == 0.2
    np.testing.assert_array_equal(best_image, img2)


def test_create_objs_with_sub_triggers_for_phase_constrain():
    phc = features.PhaseConstrain("phc")
    params = {
        "phc_phase_min": [-1.0, -0.2],
        "phc_phase_max": [1.0, 0.2],
    }

    trig_info = {
        "phc_trigger": (
            np.array([1, 2, 1, 0]),
            [
                (0, 5, 0),
                (5, 10, 1),
            ],
        )
    }

    phc.create_objs(params, trig_info)

    assert phc.f == phc.apply_trigger_seq
    assert len(phc.objs) == 3
    assert phc.objs[0].phc_phase_min == -1.0
    assert phc.objs[1].phc_phase_min == -0.2
    assert phc.objs[2].phc_phase_min == -1.0


def test_apply_trigger_seq_consumes_objects():
    phc = features.PhaseConstrain("phc")
    params = {
        "phc_phase_min": [-1.0, -0.2],
        "phc_phase_max": [1.0, 0.2],
    }

    trig_info = {
        "phc_trigger": (
            np.array([1, 2]),
            [
                (0, 5, 0),
                (5, 10, 1),
            ],
        )
    }

    phc.create_objs(params, trig_info)
    data = np.array([1 + 0j])

    phc.apply_trigger(data)
    assert len(phc.objs) == 1

    phc.apply_trigger(data)
    assert len(phc.objs) == 0


def test_low_pass_filter_apply_trigger():
    lpf = features.LowPassFilter({
        "lowpass_filter_trigger": [0, 1, 4],
        "lowpass_filter_range": [0.5, 2.0],
    })

    data = np.ones((2, 2))
    result = lpf.apply_trigger(data, 1)

    expected_sigma = lpf.filter_sigmas[1]
    np.testing.assert_array_equal(result, data + expected_sigma)


def test_pcdi_init_and_apply_partial_coherence(patch_utils):
    params = {
        "pc_type": "LUCY",
        "pc_LUCY_iterations": 5,
        "pc_normalize": True,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4), dtype=np.float32)

    pcdi = features.Pcdi(params, data)

    assert pcdi.type == "LUCY"
    assert pcdi.iterations == 5
    assert pcdi.normalize is True
    assert pcdi.kernel.shape == (2, 2)
    assert pcdi.dims == (4, 4)

    abs_amplitudes = np.ones((4, 4), dtype=np.float32)
    result = pcdi.apply_partial_coherence(abs_amplitudes)

    assert result.shape == (4, 4)


def test_pcdi_set_previous_and_update_partial_coherence(patch_utils, patch_dvut):
    params = {
        "pc_type": "LUCY",
        "pc_LUCY_iterations": 3,
        "pc_normalize": True,
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4), dtype=np.float32)

    pcdi = features.Pcdi(params, data)
    prev = np.full((4, 4), 2.0, dtype=np.float32)
    curr = np.full((4, 4), 3.0, dtype=np.float32)

    pcdi.set_previous(prev)
    pcdi.update_partial_coherence(curr)

    assert "lucy" in patch_dvut
    amplitudes2, roi_data2, kernel, iterations = patch_dvut["lucy"]
    assert iterations == 3
    assert amplitudes2.shape == (2, 2)
    assert roi_data2.shape == (2, 2)
    np.testing.assert_array_equal(pcdi.kernel, np.full((2, 2), 1.5, dtype=np.float32))


def test_pcdi_missing_iterations_raises(patch_utils):
    params = {
        "pc_LUCY_kernel": (2, 2),
    }
    data = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="pc_LUCY_iterations parameter not defined"):
        features.Pcdi(params, data)


def test_pcdi_missing_kernel_raises(patch_utils):
    params = {
        "pc_LUCY_iterations": 5,
    }
    data = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="pc_LUCY_kernel parameter not defined"):
        features.Pcdi(params, data)
# Notes
# A few implementation details in your module affect how the test is written:
#
# devlib is global
#
# The module expects set_lib() to be called before many operations.
# The fixture setup_devlib() handles that automatically for every test.
# External dependencies are mocked
#
# cohere_core.utilities.utils.crop_center
# cohere_core.utilities.utils.join
# cohere_core.utilities.dvc_utils.shrink_wrap
# cohere_core.utilities.dvc_utils.lucy_deconvolution
# Sub-trigger indexing is 1-based in the row array
#
# create_objs() converts row entries via:
# trigs = [i-1 for i in row.tolist() if i != 0]
# So in tests, row entries like [1, 2, 1] map to sub-object indices [0, 1, 0].
