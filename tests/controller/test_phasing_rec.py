# import numpy as np
# import pytest
# import cohere_core.controller.phasing as phasing
#
#
# class FakeDevLib:
#     def __init__(self):
#         self.device_set = None
#
#     def set_device(self, device_id):
#         self.device_set = device_id
#         return f"gpu:{device_id}"
#
#     def load(self, path, device=None):
#         return np.ones((4, 4, 4), dtype=np.complex64)
#
#     def from_numpy(self, arr, device=None):
#         return np.array(arr)
#
#     def fftshift(self, arr):
#         return arr
#
#     def ifftshift(self, arr):
#         return arr
#
#     def dims(self, arr):
#         return arr.shape
#
#     def copy(self, arr):
#         return np.copy(arr)
#
#     def random(self, dims, dtype=None, device=None):
#         dtype = np.dtype(dtype) if dtype is not None else np.complex64
#         if np.issubdtype(dtype, np.complexfloating):
#             return np.ones(dims, dtype=dtype) + 1j * np.ones(dims, dtype=dtype)
#         return np.ones(dims, dtype=np.float32)
#
#     def full(self, shape, fill_value, device=None):
#         return np.full(shape, fill_value)
#
#     def amax(self, arr):
#         return np.max(arr)
#
#     def absolute(self, arr):
#         return np.abs(arr)
#
#     def where(self, cond, x, y):
#         return np.where(cond, x, y)
#
#     def ifft(self, arr):
#         return arr
#
#     def fft(self, arr):
#         return arr
#
#     def hasnan(self, arr):
#         return np.isnan(arr).any()
#
#     def to_numpy(self, arr):
#         return np.array(arr)
#
#
# @pytest.fixture
# def fake_devlib(monkeypatch):
#     fake = FakeDevLib()
#     monkeypatch.setattr(phasing, "devlib", fake)
#     monkeypatch.setattr(phasing, "set_lib_from_pkg", lambda pkg: None)
#     return fake
#
#
# @pytest.fixture
# def basic_params():
#     return {"algorithm_sequence": "ER"}
#

import numpy as np
import pytest

import cohere_core.controller.phasing as phasing


def test_rec_init_sets_defaults(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")

    assert rec.params["init_guess"] == "random"
    assert rec.params["reconstructions"] == 1
    assert rec.params["hio_beta"] == 0.9
    assert rec.params["raar_beta"] == 0.45
    assert rec.params["initial_support_area"] == (0.5, 0.5, 0.5)
    assert rec.is_pcdi is False
    assert rec.phc_correction == 1


def test_rec_init_sets_ai_guess_defaults(fake_devlib):
    params = {
        "algorithm_sequence": "ER",
        "init_guess": "AI_guess",
        "shrink_wrap_threshold": 0.2,
        "shrink_wrap_gauss_sigma": 1.5,
    }

    rec = phasing.Rec(params, "data.npy", "np")

    assert rec.params["AI_threshold"] == 0.2
    assert rec.params["AI_sigma"] == 1.5


def test_rec_init_sets_twin_halves_default(fake_devlib):
    params = {
        "algorithm_sequence": "ER",
        "twin_trigger": [0, 1],
    }

    rec = phasing.Rec(params, "data.npy", "np")

    assert rec.params["twin_halves"] == (0, 0)


def test_rec_sets_pcdi_flag_when_pc_configured(fake_devlib):
    params = {
        "algorithm_sequence": "pc ER",
        "pc_interval": 5,
    }

    rec = phasing.Rec(params, "data.npy", "np")

    assert rec.is_pcdi is True


def test_init_dev_cpu_loads_npy(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")

    ret = rec.init_dev(-1)

    assert ret == 0
    assert rec.dev == "cpu"
    assert rec.dims == (4, 4, 4)


def test_init_dev_gpu_sets_device(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")

    ret = rec.init_dev(2)

    assert ret == 0
    assert rec.dev == "gpu:2"
    assert fake_devlib.device_set == 2


def test_init_dev_returns_minus_one_on_unknown_file(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.bad", "np")

    ret = rec.init_dev(-1)

    assert ret == -1


def test_init_dev_raises_in_debug_mode_on_device_error(fake_devlib, basic_params, monkeypatch):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np", debug=True)

    def raise_error(device_id):
        raise RuntimeError("device error")

    monkeypatch.setattr(fake_devlib, "set_device", raise_error)

    with pytest.raises(RuntimeError, match="device error"):
        rec.init_dev(0)


def test_init_iter_loop_creates_random_image_and_support(monkeypatch, fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.data = np.ones((4, 4, 4), dtype=np.complex64)
    rec.dims = rec.data.shape
    rec.dev = "cpu"

    monkeypatch.setattr(phasing.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(phasing.ut, "join", lambda *parts: "/".join(parts))
    monkeypatch.setattr(phasing.dvut, "pad_around", lambda center, dims, val: np.ones(dims))
    monkeypatch.setattr(phasing.dvut, "get_norm", lambda arr: 1.0)

    def fake_get_flow_arr(params, flow_items_list, gen):
        flow = np.zeros((len(flow_items_list), 1), dtype=int)
        flow[0, 0] = 1   # next
        flow[9, 0] = 1   # modulus
        flow[13, 0] = 1  # er
        return False, flow, {}

    monkeypatch.setattr(phasing.of, "get_flow_arr", fake_get_flow_arr)

    ret = rec.init_iter_loop()

    assert ret == 0
    assert rec.iter_no == 1
    assert [f.__name__ for f in rec.flow] == ["next", "modulus", "er"]
    assert rec.support.shape == (4, 4, 4)
    assert rec.ds_image.shape == (4, 4, 4)


def test_init_iter_loop_loads_existing_image_and_disables_first_run_triggers(monkeypatch, fake_devlib):
    params = {
        "algorithm_sequence": "ER",
        "lowpass_filter_trigger": [0, 1],
        "phc_trigger": [0, 1],
        "twin_trigger": [0, 1],
    }
    rec = phasing.Rec(params, "data.npy", "np")
    rec.data = np.ones((4, 4, 4), dtype=np.complex64)
    rec.dims = rec.data.shape
    rec.dev = "cpu"

    def fake_isfile(path):
        return path.endswith("image.npy") or path.endswith("support.npy")

    monkeypatch.setattr(phasing.os.path, "isfile", fake_isfile)
    monkeypatch.setattr(phasing.ut, "join", lambda *parts: "/".join(parts))

    def fake_get_flow_arr(params, flow_items_list, gen):
        flow = np.zeros((len(flow_items_list), 1), dtype=int)
        return False, flow, {}

    monkeypatch.setattr(phasing.of, "get_flow_arr", fake_get_flow_arr)

    ret = rec.init_iter_loop(dir="prev")

    assert ret == 0
    assert "lowpass_filter_trigger" not in rec.params
    assert "phc_trigger" not in rec.params
    assert "twin_trigger" not in rec.params


def test_init_iter_loop_returns_minus_one_when_flow_generation_fails(monkeypatch, fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.data = np.ones((4, 4, 4), dtype=np.complex64)
    rec.dims = rec.data.shape
    rec.dev = "cpu"

    monkeypatch.setattr(phasing.os.path, "isfile", lambda path: False)
    monkeypatch.setattr(phasing.ut, "join", lambda *parts: "/".join(parts))
    monkeypatch.setattr(phasing.dvut, "pad_around", lambda center, dims, val: np.ones(dims))
    monkeypatch.setattr(phasing.dvut, "get_norm", lambda arr: 1.0)

    def raise_error(params, flow_items_list, gen):
        raise RuntimeError("bad flow")

    monkeypatch.setattr(phasing.of, "get_flow_arr", raise_error)

    ret = rec.init_iter_loop()

    assert ret == -1


def test_getters_return_expected_values(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[1]])
    rec.support = np.array([[2]])
    rec.is_pc = False

    assert np.array_equal(rec.get_image(), np.array([[1]]))
    assert np.array_equal(rec.get_support(), np.array([[2]]))
    assert rec.get_coherence() is None
