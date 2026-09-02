import numpy as np
import pytest

import cohere_core.controller.phasing as phasing


def test_next_increments_iteration(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.iter = -1

    rec.next()

    assert rec.iter == 0


def test_get_ratio_masks_small_divisors(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")

    dividend = np.array([2.0, 4.0, 6.0])
    divisor = np.array([1.0, 0.0, 1e-12])

    result = rec.get_ratio(dividend, divisor)

    np.testing.assert_allclose(result, np.array([2.0, 0.0, 0.0]))


def test_lowpass_filter_operation_calls_trigger(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.data = np.array([1, 2, 3])
    rec.iter = 7

    class Obj:
        def apply_trigger(self, data, iteration):
            assert iteration == 7
            return data * 2

    rec.lowpass_filter_obj = Obj()
    rec.lowpass_filter_operation()

    np.testing.assert_array_equal(rec.iter_data, np.array([2, 4, 6]))


def test_reset_resolution_restores_data(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.data = np.array([5, 6, 7])
    rec.iter_data = np.array([1, 1, 1])

    rec.reset_resolution()

    np.testing.assert_array_equal(rec.iter_data, rec.data)


def test_shrink_wrap_operation_updates_support(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[1, 2]])

    class Obj:
        def apply_trigger(self, image):
            return image > 1

    rec.shrink_wrap_obj = Obj()
    rec.shrink_wrap_operation()

    np.testing.assert_array_equal(rec.support, np.array([[False, True]]))


def test_reset_phc_correction_sets_one(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.phc_correction = np.array([2])

    rec.reset_phc_correction()

    assert rec.phc_correction == 1


def test_phc_operation_updates_correction(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[1, 2]])

    class Obj:
        def apply_trigger(self, image):
            return image * 3

    rec.phc_obj = Obj()
    rec.phc_operation()

    np.testing.assert_array_equal(rec.phc_correction, np.array([[3, 6]]))


def test_to_reciprocal_space_uses_ifft(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[1, 2]])

    rec.to_reciprocal_space()

    np.testing.assert_array_equal(rec.rs_amplitudes, rec.ds_image)


def test_pc_operation_updates_partial_coherence(fake_devlib):
    params = {"algorithm_sequence": "pc ER", "pc_interval": 1}
    rec = phasing.Rec(params, "data.npy", "np")
    rec.rs_amplitudes = np.array([[3.0, 4.0]])

    called = {}

    class Obj:
        def update_partial_coherence(self, amplitudes):
            called["arg"] = amplitudes

    rec.pc_obj = Obj()
    rec.pc_operation()

    np.testing.assert_array_equal(called["arg"], np.array([[3.0, 4.0]]))


def test_pc_modulus_updates_error_and_scales_amplitudes(fake_devlib, monkeypatch):
    params = {"algorithm_sequence": "pc ER", "pc_interval": 1}
    rec = phasing.Rec(params, "data.npy", "np")
    rec.iter_data = np.array([[2.0, 2.0]])
    rec.rs_amplitudes = np.array([[1.0, 2.0]])
    rec.errs = []

    monkeypatch.setattr(phasing.dvut, "get_norm", lambda arr: np.linalg.norm(arr))

    class Obj:
        def apply_partial_coherence(self, abs_amplitudes):
            return np.array([[1.0, 1.0]])

    rec.pc_obj = Obj()

    rec.pc_modulus()

    assert len(rec.errs) == 1
    np.testing.assert_allclose(rec.rs_amplitudes, np.array([[2.0, 4.0]]))


def test_modulus_updates_error_and_scales_amplitudes(fake_devlib, monkeypatch, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.iter_data = np.array([[2.0, 4.0]])
    rec.rs_amplitudes = np.array([[1.0, 2.0]])
    rec.errs = []

    monkeypatch.setattr(phasing.dvut, "get_norm", lambda arr: np.linalg.norm(arr))

    rec.modulus()

    assert len(rec.errs) == 1
    np.testing.assert_allclose(rec.rs_amplitudes, np.array([[2.0, 4.0]]))


def test_global_min_operation_calls_trigger(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[1.0]])
    rec.errs = [0.123]

    called = {}

    class Obj:
        def apply_trigger(self, image, err):
            called["image"] = image
            called["err"] = err

    rec.global_min_obj = Obj()
    rec.global_min_operation()

    assert called["err"] == 0.123
    np.testing.assert_array_equal(called["image"], rec.ds_image)


def test_set_prev_pc_uses_absolute_rs_amplitudes(fake_devlib):
    params = {"algorithm_sequence": "pc ER", "pc_interval": 1}
    rec = phasing.Rec(params, "data.npy", "np")
    rec.rs_amplitudes = np.array([[3.0, 4.0]])

    called = {}

    class Obj:
        def set_previous(self, arr):
            called["arr"] = arr

    rec.pc_obj = Obj()
    rec.set_prev_pc()

    np.testing.assert_array_equal(called["arr"], np.array([[3.0, 4.0]]))


def test_to_direct_space_uses_fft(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.rs_amplitudes = np.array([[1, 2]])

    rec.to_direct_space()

    np.testing.assert_array_equal(rec.ds_image_proj, np.array([[1, 2]]))


def test_er_applies_support_and_phc(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image_proj = np.array([[1, 2], [3, 4]], dtype=float)
    rec.support = np.array([[1, 0], [1, 0]], dtype=float)
    rec.phc_correction = np.array([[1, 1], [0, 1]], dtype=float)

    rec.er()

    expected = np.array([[1, 0], [0, 0]], dtype=float)
    np.testing.assert_allclose(rec.ds_image, expected)


def test_hio_inside_and_outside_support(fake_devlib):
    rec = phasing.Rec({"algorithm_sequence": "HIO", "hio_beta": 0.5}, "data.npy", "np")
    rec.ds_image = np.array([[10.0, 20.0], [30.0, 40.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0], [3.0, 4.0]])
    rec.support = np.array([[1, 0], [1, 0]])
    rec.phc_correction = np.ones((2, 2))

    rec.hio()

    expected = np.array([
        [1.0, 19.0],
        [3.0, 38.0],
    ])
    np.testing.assert_allclose(rec.ds_image, expected)


def test_sf_operation(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image_proj = np.array([[1.0, 2.0], [3.0, 4.0]])
    rec.support = np.array([[1.0, 0.0], [1.0, 0.0]])
    rec.phc_correction = 1

    rec.sf()

    expected = 2.0 * (rec.ds_image_proj * rec.support) - rec.ds_image_proj
    np.testing.assert_allclose(rec.ds_image, expected)


def test_raar_operation(fake_devlib):
    rec = phasing.Rec({"algorithm_sequence": "RAAR", "raar_beta": 0.25}, "data.npy", "np")
    rec.ds_image = np.array([[10.0, 20.0], [30.0, 40.0]])
    rec.ds_image_proj = np.array([[1.0, 2.0], [3.0, 4.0]])
    rec.support = np.array([[1.0, 0.0], [1.0, 0.0]])
    rec.phc_correction = 1

    rec.raar()

    expected = (
        0.25 * (rec.support * rec.ds_image_proj + np.array([[10.0, 20.0], [30.0, 40.0]]))
        + 0.5 * rec.ds_image_proj
    )
    np.testing.assert_allclose(rec.ds_image, expected)


def test_twin_operation_zeroes_expected_halves(fake_devlib):
    rec = phasing.Rec(
        {"algorithm_sequence": "ER", "twin_trigger": [0, 1], "twin_halves": (0, 1)},
        "data.npy",
        "np",
    )
    rec.ds_image = np.ones((4, 4, 2), dtype=float)

    rec.twin_operation()

    assert np.all(rec.ds_image[2:, :, :] == 0)
    assert np.all(rec.ds_image[:, :2, :] == 0)


def test_average_operation_initializes_and_accumulates(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")

    rec.ds_image = np.array([[3.0, 4.0]])
    rec.average_operation()
    np.testing.assert_allclose(rec.aver, np.array([[3.0, 4.0]]))
    assert rec.aver_iter == 1

    rec.ds_image = np.array([[6.0, 8.0]])
    rec.average_operation()
    np.testing.assert_allclose(rec.aver, np.array([[9.0, 12.0]]))
    assert rec.aver_iter == 2


def test_iterate_runs_flow_and_normalizes_result(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.ones((2, 2), dtype=np.complex64)
    rec.errs = []

    def f1():
        rec.iter = 0

    def f2():
        rec.errs.append(0.5)

    def f3():
        rec.ds_image = rec.ds_image * 10

    rec.flow = [f1, f2, f3]

    ret = rec.iterate()

    assert ret == 0
    assert rec.iter == 0
    assert rec.errs == [0.5]
    np.testing.assert_allclose(rec.ds_image, np.ones((2, 2)))


def test_iterate_returns_minus_one_on_exception_when_not_debug(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np", debug=False)
    rec.ds_image = np.ones((2, 2))
    rec.flow = [lambda: (_ for _ in ()).throw(RuntimeError("boom"))]

    assert rec.iterate() == -1


def test_iterate_raises_on_exception_when_debug(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np", debug=True)
    rec.ds_image = np.ones((2, 2))
    rec.flow = [lambda: (_ for _ in ()).throw(RuntimeError("boom"))]

    with pytest.raises(RuntimeError, match="boom"):
        rec.iterate()


def test_iterate_returns_minus_one_on_nan(fake_devlib, basic_params):
    rec = phasing.Rec(dict(basic_params), "data.npy", "np")
    rec.ds_image = np.array([[np.nan]])
    rec.flow = []

    assert rec.iterate() == -1