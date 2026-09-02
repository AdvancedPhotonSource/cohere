import os
import numpy as np
import pytest

import cohere_core.utilities.ga_utils as ga_utils


@pytest.fixture
def patch_ut_join(monkeypatch):
    """
    Patch ga_utils.ut.join to os.path.join so tests don't depend on external utility behavior.
    """
    monkeypatch.setattr(ga_utils.ut, "join", os.path.join)


def test_read_results_all_files_exist(tmp_path, patch_ut_join):
    arr_image = np.array([[1, 2], [3, 4]])
    arr_support = np.array([[0, 1], [1, 0]])
    arr_coh = np.array([[5, 6], [7, 8]])

    np.save(tmp_path / "image.npy", arr_image)
    np.save(tmp_path / "support.npy", arr_support)
    np.save(tmp_path / "coherence.npy", arr_coh)

    image, support, coh = ga_utils.read_results(tmp_path)

    np.testing.assert_array_equal(image, arr_image)
    np.testing.assert_array_equal(support, arr_support)
    np.testing.assert_array_equal(coh, arr_coh)


def test_read_results_missing_files_returns_none(tmp_path, patch_ut_join):
    arr_image = np.array([1, 2, 3])
    np.save(tmp_path / "image.npy", arr_image)

    image, support, coh = ga_utils.read_results(tmp_path)

    np.testing.assert_array_equal(image, arr_image)
    assert support is None
    assert coh is None


def test_read_results_no_files_returns_all_none(tmp_path, patch_ut_join):
    image, support, coh = ga_utils.read_results(tmp_path)

    assert image is None
    assert support is None
    assert coh is None


def test_tracing_random_init_guess(patch_ut_join):
    pars = {"init_guess": "random"}
    tracing = ga_utils.Tracing(reconstructions=3, pars=pars, dir="/base")

    assert tracing.init_dirs == [None, None, None]
    assert tracing.report_tracing == [["random0"], ["random1"], ["random2"]]


def test_tracing_ai_guess(patch_ut_join):
    pars = {"init_guess": "AI_guess"}
    tracing = ga_utils.Tracing(reconstructions=3, pars=pars, dir="/base")

    assert tracing.init_dirs == [os.path.join("/base", "results_AI"), None, None]
    assert tracing.report_tracing == [["AI_guess"], ["random0"], ["random1"]]


def test_tracing_continue_init_guess_uses_valid_subdirs(tmp_path, patch_ut_join):
    continue_dir = tmp_path / "continue"
    continue_dir.mkdir()

    valid1 = continue_dir / "run1"
    valid1.mkdir()
    np.save(valid1 / "image.npy", np.array([1]))

    valid2 = continue_dir / "run2"
    valid2.mkdir()
    np.save(valid2 / "image.npy", np.array([2]))

    invalid = continue_dir / "run3"
    invalid.mkdir()
    # no image.npy here

    pars = {
        "init_guess": "continue",
        "continue_dir": str(continue_dir),
    }

    tracing = ga_utils.Tracing(reconstructions=3, pars=pars, dir="/base")

    expected_dirs = [
        os.path.join(str(continue_dir), "run1"),
        os.path.join(str(continue_dir), "run2"),
        None,
    ]

    assert tracing.init_dirs == expected_dirs
    assert tracing.report_tracing[0] == [os.path.join(str(continue_dir), "run1")]
    assert tracing.report_tracing[1] == [os.path.join(str(continue_dir), "run2")]
    assert tracing.report_tracing[2] == ["random0"]


def test_append_gen_uses_map_indices(patch_ut_join):
    pars = {"init_guess": "random"}
    tracing = ga_utils.Tracing(reconstructions=2, pars=pars, dir="/base")
    tracing.set_map({"a": 1, "b": 0})

    gen_ranks = {
        "a": ("indA", {"chi": 0.1}),
        "b": ("indB", {"chi": 0.2}),
    }
    tracing.append_gen(gen_ranks)

    assert tracing.report_tracing[0] == ["random0", ("indB", {"chi": 0.2})]
    assert tracing.report_tracing[1] == ["random1", ("indA", {"chi": 0.1})]


def test_pretty_format_results_contains_expected_content(patch_ut_join):
    pars = {"init_guess": "random"}
    tracing = ga_utils.Tracing(reconstructions=2, pars=pars, dir="/base")
    tracing.set_map({"k0": 0, "k1": 1})

    tracing.append_gen({
        "k0": ("child0", {"chi": 0.1, "sharpness": 10}),
        "k1": ("child1", {"chi": 0.2, "sharpness": 20}),
    })

    output = tracing.pretty_format_results()

    assert "start" in output
    assert "generation 0" in output
    assert "random0" in output
    assert "random1" in output
    assert "child0" in output
    assert "child1" in output
    assert "chi : 0.1" in output
    assert "sharpness : 20" in output


def test_save_writes_pretty_report(tmp_path, patch_ut_join):
    pars = {"init_guess": "random"}
    tracing = ga_utils.Tracing(reconstructions=1, pars=pars, dir="/base")
    tracing.set_map({"k0": 0})
    tracing.append_gen({"k0": ("child0", {"chi": 0.1})})

    tracing.save(tmp_path)

    report_file = tmp_path / "ranks.txt"
    assert report_file.exists()

    text = report_file.read_text()
    assert "child0" in text
    assert "chi : 0.1" in text


def test_save_falls_back_to_raw_format_when_pretty_format_fails(tmp_path, patch_ut_join, monkeypatch):
    pars = {"init_guess": "random"}
    tracing = ga_utils.Tracing(reconstructions=1, pars=pars, dir="/base")

    def broken():
        raise RuntimeError("formatting failed")

    monkeypatch.setattr(tracing, "pretty_format_results", broken)

    tracing.save(tmp_path)

    report_file = tmp_path / "ranks.txt"
    assert report_file.exists()

    text = report_file.read_text()
    assert "random0" in text


def test_set_map_sets_internal_map(patch_ut_join):
    tracing = ga_utils.Tracing(reconstructions=1, pars={"init_guess": "random"}, dir="/base")
    mapping = {"x": 0}
    tracing.set_map(mapping)
    assert tracing.map == mapping


def test_set_ga_defaults_populates_defaults():
    pars = {
        "algorithm_sequence": [],
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result["reconstructions"] == 1
    assert result["ga_generations"] == 1
    assert result["init_guess"] == "random"
    assert result["ga_fast"] is False
    assert result["ga_metrics"] == ["chi"]
    assert result["worst_remove_no"] == [0]
    assert result["ga_reconstructions"] == [1]
    assert result["ga_sw_thresholds"] == [0.1]
    assert result["ga_sw_gauss_sigmas"] == [1.0]
    assert result["ga_breed_modes"] == ["sqrt_ab"]
    assert result["low_resolution_generations"] == 0


def test_set_ga_defaults_expands_single_value_lists():
    pars = {
        "algorithm_sequence": [],
        "ga_generations": 3,
        "reconstructions": 5,
        "ga_metrics": ["sharpness"],
        "ga_sw_thresholds": [0.2],
        "ga_sw_gauss_sigmas": [2.0],
        "ga_breed_modes": ["avg_ab"],
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result["ga_metrics"] == ["sharpness", "sharpness", "sharpness"]
    assert result["ga_sw_thresholds"] == [0.2, 0.2, 0.2]
    assert result["ga_sw_gauss_sigmas"] == [2.0, 2.0, 2.0]
    assert result["ga_breed_modes"] == ["avg_ab", "avg_ab", "avg_ab"]


def test_set_ga_defaults_pads_short_lists():
    pars = {
        "algorithm_sequence": [],
        "ga_generations": 3,
        "reconstructions": 4,
        "ga_metrics": ["chi", "sharpness"],
        "ga_cullings": [1],
        "ga_sw_thresholds": [0.3, 0.4],
        "ga_sw_gauss_sigmas": [1.5, 2.5],
        "ga_breed_modes": ["sqrt_ab", "avg_ab"],
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result["ga_metrics"] == ["chi", "sharpness", "chi"]
    assert result["worst_remove_no"] == [1, 0, 0]
    assert result["ga_reconstructions"] == [3, 3, 3]
    assert result["ga_sw_thresholds"] == [0.3, 0.4, 0.1]
    assert result["ga_sw_gauss_sigmas"] == [1.5, 2.5, 1.0]
    assert result["ga_breed_modes"] == ["sqrt_ab", "avg_ab", "sqrt_ab"]


def test_set_ga_defaults_returns_error_when_culled_to_zero():
    pars = {
        "algorithm_sequence": [],
        "reconstructions": 2,
        "ga_generations": 2,
        "ga_cullings": [1, 1],
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result == "culled down to 0 reconstructions, check configuration"


def test_set_ga_defaults_sets_low_resolution_fields():
    pars = {
        "algorithm_sequence": [],
        "ga_generations": 2,
        "reconstructions": 3,
        "ga_lpf_sigmas": [1.0, 2.0],
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result["low_resolution_generations"] == 2
    assert result["low_resolution_alg"] == "GAUSS"


def test_set_ga_defaults_preserves_existing_low_resolution_alg():
    pars = {
        "algorithm_sequence": [],
        "ga_generations": 2,
        "reconstructions": 3,
        "ga_lpf_sigmas": [1.0],
        "low_resolution_alg": "CUSTOM",
    }

    result = ga_utils.set_ga_defaults(pars)

    assert result["low_resolution_generations"] == 1
    assert result["low_resolution_alg"] == "CUSTOM"


def test_set_ga_defaults_pc_settings():
    pars = {
        "algorithm_sequence": ["er", "pc"],
        "pc_interval": 5,
        "ga_generations": 3,
        "reconstructions": 2,
    }

    result = ga_utils.set_ga_defaults(pars)

    assert "ga_gen_pc_start" in result
    assert result["ga_gen_pc_start"] == 0


# What these tests cover
# read_results
# all files present
# some files missing
# all files missing
# Tracing
# random init mode
# AI_guess init mode
# continue mode
# appending generation results
# formatting output
# saving output
# fallback behavior if formatting fails
# set_map
# set_ga_defaults
# setting defaults
# expanding single-item lists
# padding short lists
# handling culling to zero
# low-resolution config
# pc-related config
# Run the tests
#
# pytest -q
# Small note about one implementation detail
# In Tracing.pretty_format_results, this line:
#
#
# fitnesses = list(self.report_tracing[0][1][1].keys())
# assumes that at least one generation has already been appended. So if you call save() before append_gen(), pretty formatting will fail and your code will intentionally fall back to raw formatting. I included a test for that fallback path.
