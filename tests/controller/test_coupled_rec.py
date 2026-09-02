import numpy as np
import pytest

import cohere_core.controller.phasing as phasing


class FakePeak:
    def __init__(self, hkl=(1, 0, 0), total=1.0):
        self.hkl = hkl
        self.total = total
        self.weight = 1.0
        self.conf_hist = []
        self.conf_iter = []
        self.weight_iter = 0
        self.res_data = np.ones((4, 4, 4))
        self.data = np.ones((4, 4, 4))
        self.window = np.ones((4, 4, 4), dtype=bool)
        self.mask = np.zeros((4, 4, 4), dtype=bool)
        self.filter = np.zeros((4, 4, 4), dtype=bool)
        self.window_size = 1.0
        self.g_vec = np.array([1.0, 0.0, 0.0])
        self.gdotg = np.array(1.0)

    def normalize(self, norm):
        self.total = norm


def coupled_params():
    return {
        "algorithm_sequence": "ER",
        "adapt_alien_threshold": 2.0,
        "adapt_threshold_init": 0.5,
        "adapt_threshold_iters": [],
        "adapt_threshold_vals": [],
        "weight_init": 0.1,
        "weight_iters": [],
        "weight_vals": [],
        "adapt_power": 2.0,
        "adapt_alien_start": 999,
        "calc_strain": False,
    }


def test_coupled_rec_init_sets_defaults(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["peak1", "peak2"], "np")

    assert rec.params["switch_peak_trigger"] == [0, 5]
    assert rec.params["adapt_trigger"] == []
    assert rec.params["calc_strain"] is False
    assert rec.er_iter is False


def test_coupled_init_dev_initializes_peaks(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing, "Peak", lambda d: FakePeak(hkl=(1, 0, 0) if d == "p1" else (0, 1, 0), total=2.0))

    rec = phasing.CoupledRec(coupled_params(), ["p1", "p2"], "np")
    ret = rec.init_dev(-1)

    assert ret == 0
    assert rec.num_peaks == 2
    assert rec.pk == 0
    assert rec.data.shape == (4, 4, 4)
    assert rec.iter_data.shape == (4, 4, 4)
    assert rec.window.shape == (4, 4, 4)


def test_coupled_init_dev_respects_control_peak(monkeypatch, fake_devlib):
    monkeypatch.setattr(
        phasing,
        "Peak",
        lambda d: FakePeak(hkl=(1, 0, 0) if d == "p1" else (0, 1, 0), total=2.0),
    )

    params = coupled_params()
    params["control_peak"] = (0, 1, 0)

    rec = phasing.CoupledRec(params, ["p1", "p2"], "np")
    ret = rec.init_dev(-1)

    assert ret == 0
    assert len(rec.peak_objs) == 1
    assert rec.ctrl_peak is not None
    assert rec.ctrl_error == []


def test_coupled_init_iter_loop_sets_shared_images(monkeypatch, fake_devlib):
    monkeypatch.setattr(phasing.Rec, "init_iter_loop", lambda self, img_dir=None, gen=None: 0)

    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.ds_image = np.ones((4, 4, 4), dtype=np.complex64)
    rec.iter_no = 3

    ret = rec.init_iter_loop()

    assert ret == 0
    assert rec.rho_image.shape == (4, 4, 4)
    assert rec.u_image.shape == (4, 4, 4, 3)
    assert len(rec.proj_weight) == 3
    assert len(rec.peak_threshold) == 3


def test_adapt_operation_turns_on_adaptation(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.adapt_on = False

    rec.adapt_operation()

    assert rec.adapt_on is True


def test_update_weights_uses_relative_confidence(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1", "p2"], "np")
    p1 = FakePeak()
    p2 = FakePeak(hkl=(0, 1, 0))
    p1.conf_hist = [0.4, 0.6, 0.5]
    p2.conf_hist = [1.0, 0.8, 0.9]
    rec.peak_objs = [p1, p2]
    rec.params["adapt_power"] = 2.0
    rec.adapt_on = True

    rec.update_weights()

    assert pytest.approx(p2.weight) == 1.0
    assert p1.weight < p2.weight
    assert rec.adapt_on is False


def test_switch_peak_operation_advances_peak(monkeypatch, fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1", "p2"], "np")
    rec.num_peaks = 2
    rec.pk = 0
    rec.iter = 1
    rec.peak_objs = [FakePeak(), FakePeak(hkl=(0, 1, 0))]
    rec.peak_weights = []
    rec.iter_weights = []
    rec.adapt_on = False

    called = {"shared": 0, "working": 0, "control": 0}
    monkeypatch.setattr(rec, "to_shared_image", lambda: called.__setitem__("shared", called["shared"] + 1))
    monkeypatch.setattr(rec, "to_working_image", lambda: called.__setitem__("working", called["working"] + 1))
    monkeypatch.setattr(rec, "get_control_error", lambda: called.__setitem__("control", called["control"] + 1))

    rec.switch_peak_operation()

    assert called["shared"] == 1
    assert called["working"] == 1
    assert called["control"] == 1
    assert rec.pk == 1
    assert len(rec.peak_weights) == 1
    assert len(rec.iter_weights) == 1


# def test_to_working_image_builds_projected_image(fake_devlib):
#     from unittest.mock import patch
#     with patch.object(phasing, "devlib", np, create=True):
#         rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
#         rec.peak_objs = [FakePeak()]
#         rec.pk = 0
#         rec.rho_image = np.ones((4, 4, 4))
#         rec.u_image = np.zeros((4, 4, 4, 3))
#         rec.support = np.ones((4, 4, 4))
#         rec.iter = 0
#         rec.iter_no = 500
#
#         rec.to_working_image()
#
#         assert rec.ds_image.shape == (4, 4, 4)
#         assert rec.proj.shape == (4, 4, 4)
#         assert rec.iter_data.shape == (4, 4, 4)


def test_to_shared_image_updates_density_and_displacement(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.peak_objs = [FakePeak()]
    rec.pk = 0
    rec.iter = 0
    rec.proj_weight = np.array([0.5])
    rec.rho_image = np.ones((4, 4, 4))
    rec.u_image = np.zeros((4, 4, 4, 3))
    rec.ds_image = np.ones((4, 4, 4), dtype=np.complex64)
    rec.support = np.ones((4, 4, 4))
    rec.dims = (4, 4, 4)
    rec.er_iter = False

    rec.to_shared_image()

    assert rec.rho_image.shape == (4, 4, 4)
    assert rec.u_image.shape == (4, 4, 4, 3)


def test_coupled_er_sets_flag_and_applies_support(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 0.0]])

    rec.er()

    assert rec.er_iter is True
    np.testing.assert_allclose(rec.ds_image, np.array([[1.0, 0.0]]))


def test_coupled_hio_sets_flag_and_applies_update(fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.params["hio_beta"] = 0.5
    rec.ds_image = np.array([[10.0, 20.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0]])
    rec.support = np.array([[1.0, 0.0]])

    rec.hio()

    assert rec.er_iter is False
    np.testing.assert_allclose(rec.ds_image, np.array([[1.0, 19.0]]))


def test_coupled_shrink_wrap_skips_during_er(monkeypatch, fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.er_iter = True
    rec.pk = 0
    rec.iter = 0
    rec.peak_objs = [FakePeak()]
    rec.peak_threshold = np.array([0.5])

    called = {"super": 0}
    monkeypatch.setattr(phasing.Rec, "shrink_wrap_operation", lambda self: called.__setitem__("super", 1))

    rec.shrink_wrap_operation()

    assert called["super"] == 0


def test_coupled_shrink_wrap_skips_for_low_weight(monkeypatch, fake_devlib):
    rec = phasing.CoupledRec(coupled_params(), ["p1"], "np")
    rec.er_iter = False
    rec.pk = 0
    rec.iter = 0
    peak = FakePeak()
    peak.weight = 0.1
    rec.peak_objs = [peak]
    rec.peak_threshold = np.array([0.5])

    called = {"super": 0}
    monkeypatch.setattr(phasing.Rec, "shrink_wrap_operation", lambda self: called.__setitem__("super", 1))

    rec.shrink_wrap_operation()

    assert called["super"] == 0