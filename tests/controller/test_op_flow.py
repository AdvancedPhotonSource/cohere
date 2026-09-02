# test_op_flow.py

import numpy as np
import pytest

from cohere_core.controller import op_flow


def test_get_alg_rows_simple_sequence():
    s = "2*ER+3*HIO"
    alg_rows, sub_rows, iter_no, pc_start = op_flow.get_alg_rows(s, pc_conf_start=None)

    assert iter_no == 5
    assert pc_start is None
    assert sub_rows == {}

    np.testing.assert_array_equal(
        alg_rows["to_reciprocal_space"],
        np.array([1, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(
        alg_rows["modulus"],
        np.array([1, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(
        alg_rows["to_direct_space"],
        np.array([1, 1, 1, 1, 1])
    )
    np.testing.assert_array_equal(
        alg_rows["er"],
        np.array([1, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        alg_rows["hio"],
        np.array([0, 0, 1, 1, 1])
    )


def test_get_alg_rows_grouped_sequence():
    s = "2*(1*ER+1*HIO)+1*ER"
    alg_rows, sub_rows, iter_no, pc_start = op_flow.get_alg_rows(s, pc_conf_start=None)

    assert iter_no == 5
    assert pc_start is None
    assert sub_rows == {}

    np.testing.assert_array_equal(
        alg_rows["er"],
        np.array([1, 0, 1, 0, 1])
    )
    np.testing.assert_array_equal(
        alg_rows["hio"],
        np.array([0, 1, 0, 1, 0])
    )


def test_get_alg_rows_with_sub_triggers():
    s = "3*ER.SW0+2*HIO.PHC1"
    alg_rows, sub_rows, iter_no, pc_start = op_flow.get_alg_rows(s, pc_conf_start=None)

    assert iter_no == 5
    assert pc_start is None

    assert "shrink_wrap_trigger" in sub_rows
    assert "phc_trigger" in sub_rows

    assert sub_rows["shrink_wrap_trigger"] == [(0, 3, "0")]
    assert sub_rows["phc_trigger"] == [(3, 5, "1")]


def test_get_alg_rows_invalid_algorithm():
    s = "2*BADALG"
    with pytest.raises(NameError, match="algorithm BADALG is not defined"):
        op_flow.get_alg_rows(s, pc_conf_start=None)


def test_get_alg_rows_partial_coherence_start_true():
    s = "2*ERpc+2*HIO"
    alg_rows, sub_rows, iter_no, pc_start = op_flow.get_alg_rows(s, pc_conf_start=True)

    assert iter_no == 4
    assert pc_start == 0

    np.testing.assert_array_equal(
        alg_rows["pc_modulus"],
        np.array([1, 1, 0, 0])
    )
    np.testing.assert_array_equal(
        alg_rows["modulus"],
        np.array([0, 0, 1, 1])
    )


def test_get_alg_rows_partial_coherence_disabled():
    s = "2*ERpc+2*HIOpc"
    alg_rows, sub_rows, iter_no, pc_start = op_flow.get_alg_rows(s, pc_conf_start=None)

    assert iter_no == 4
    assert pc_start is None

    np.testing.assert_array_equal(
        alg_rows["pc_modulus"],
        np.array([0, 0, 0, 0])
    )
    np.testing.assert_array_equal(
        alg_rows["modulus"],
        np.array([1, 1, 1, 1])
    )


def test_fill_trigger_row_single_index():
    row = op_flow.fill_trigger_row([2], iter_no=5, last_trig=False)
    np.testing.assert_array_equal(row, np.array([0, 0, 1, 0, 0]))


def test_fill_trigger_row_single_negative_index():
    row = op_flow.fill_trigger_row([-1], iter_no=5, last_trig=False)
    np.testing.assert_array_equal(row, np.array([0, 0, 0, 0, 1]))


def test_fill_trigger_row_range_without_last_trig():
    row = op_flow.fill_trigger_row([1, 2, 5], iter_no=6, last_trig=False)
    np.testing.assert_array_equal(row, np.array([0, 1, 0, 1, 0, 0]))


def test_fill_trigger_row_range_with_last_trig():
    row = op_flow.fill_trigger_row([1, 2, 5], iter_no=6, last_trig=True)
    np.testing.assert_array_equal(row, np.array([0, 1, 0, 1, 1, 0]))


def test_fill_sub_trigger_row_basic():
    sub_iters = [(0, 3, "0"), (3, 6, "1")]
    sub_trigs = [
        [0, 1],   # applies to first chunk -> iterations 0,1,2
        [0, 2],   # applies to second chunk -> iterations 3,5
    ]

    row = op_flow.fill_sub_trigger_row(sub_iters, sub_trigs, iter_no=6, last_trig=False)

    # first chunk uses index 0 -> stored as 1
    # second chunk uses index 1 -> stored as 2
    np.testing.assert_array_equal(row, np.array([1, 1, 1, 2, 0, 2]))


def test_fill_sub_trigger_row_not_enough_entries():
    sub_iters = [(0, 3, "1")]
    sub_trigs = [[0, 1]]  # only index 0 exists, but sub_iter requests index 1

    with pytest.raises(RuntimeError, match="not enough entries in sub-trigger"):
        op_flow.fill_sub_trigger_row(sub_iters, sub_trigs, iter_no=3, last_trig=False)


def test_get_flow_arr_basic_integration():
    params = {
        "algorithm_sequence": "2*ER+2*HIO",
        "progress_trigger": [0, 1, 4],
    }
    flow_items = [
        "next",
        "to_reciprocal_space",
        "modulus",
        "to_direct_space",
        "er",
        "hio",
        "progress_operation",
    ]

    has_pc, flow_arr, sub_trig_op = op_flow.get_flow_arr(params, flow_items)

    assert has_pc is False
    assert sub_trig_op == {}
    assert flow_arr.shape == (len(flow_items), 4)

    np.testing.assert_array_equal(flow_arr[0], np.array([1, 1, 1, 1]))  # next
    np.testing.assert_array_equal(flow_arr[1], np.array([1, 1, 1, 1]))  # to_reciprocal_space
    np.testing.assert_array_equal(flow_arr[2], np.array([1, 1, 1, 1]))  # modulus
    np.testing.assert_array_equal(flow_arr[3], np.array([1, 1, 1, 1]))  # to_direct_space
    np.testing.assert_array_equal(flow_arr[4], np.array([1, 1, 0, 0]))  # er
    np.testing.assert_array_equal(flow_arr[5], np.array([0, 0, 1, 1]))  # hio
    np.testing.assert_array_equal(flow_arr[6], np.array([1, 1, 1, 1]))  # progress_operation


def test_get_flow_arr_with_sub_trigger_operation():
    params = {
        "algorithm_sequence": "3*ER.SW0+2*HIO",
        "shrink_wrap_trigger": [
            [0, 1],  # for SW0 chunk, trigger every iteration in that chunk
        ],
    }
    flow_items = [
        "to_reciprocal_space",
        "modulus",
        "to_direct_space",
        "er",
        "hio",
        "shrink_wrap_operation",
    ]

    has_pc, flow_arr, sub_trig_op = op_flow.get_flow_arr(params, flow_items)

    assert has_pc is False
    assert "shrink_wrap_trigger" in sub_trig_op
    assert flow_arr.shape == (len(flow_items), 5)

    np.testing.assert_array_equal(flow_arr[3], np.array([1, 1, 1, 0, 0]))  # er
    np.testing.assert_array_equal(flow_arr[4], np.array([0, 0, 0, 1, 1]))  # hio
    np.testing.assert_array_equal(flow_arr[5], np.array([1, 1, 1, 0, 0]))  # shrink_wrap_operation


def test_get_flow_arr_partial_coherence_sets_pc_trigger():
    params = {
        "algorithm_sequence": "2*ER+2*HIOpc",
        "pc_interval": 1,
    }
    flow_items = [
        "modulus",
        "pc_operation",
        "set_prev_pc",
    ]

    has_pc, flow_arr, sub_trig_op = op_flow.get_flow_arr(params, flow_items)

    assert has_pc is True
    assert "pc_trigger" in params
    assert params["pc_trigger"] == [2, 1]

    # pc starts at iteration 2
    np.testing.assert_array_equal(flow_arr[1], np.array([0, 0, 1, 1]))
    # set_prev_pc is shifted one iteration earlier
    np.testing.assert_array_equal(flow_arr[2], np.array([0, 1, 1, 0]))


def test_get_flow_arr_returns_false_on_bad_sequence():
    params = {
        "algorithm_sequence": "2*NOTREAL",
    }

    with pytest.raises(NameError, match="algorithm NOTREAL is not defined in op_flow.py file, algs dict."):
        op_flow.get_alg_rows(params["algorithm_sequence"], pc_conf_start=None)


def test_get_flow_arr_reset_resolution_from_lowpass_filter():
    params = {
        "algorithm_sequence": "5*ER",
        "lowpass_filter_trigger": [0, 1, 3],
    }
    flow_items = [
        "modulus",
        "reset_resolution",
    ]

    has_pc, flow_arr, sub_trig_op = op_flow.get_flow_arr(params, flow_items)

    assert has_pc is False
    np.testing.assert_array_equal(flow_arr[1], np.array([0, 0, 0, 1, 0]))

