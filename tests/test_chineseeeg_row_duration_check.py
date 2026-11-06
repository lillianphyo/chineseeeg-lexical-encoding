import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location("row_check", Path(__file__).parents[1] / "scripts" / "chineseeeg_row_duration_check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _rows(overhead, n_values=(2, 3, 5, 8, 10), reps=20, seed=0):
    rng = np.random.default_rng(seed)
    n = np.repeat(n_values, reps)
    dur = n * check.SHIFT_TIME + (n - 1) * overhead + rng.normal(0, 0.005, len(n))
    return pd.DataFrame({"n": n, "dur": dur})


def test_fit_recovers_the_rebuild_overhead_from_slope_and_intercept():
    fit = check.fit_row_model(_rows(overhead=0.08))
    assert fit["slope"] == pytest.approx(0.43, abs=0.005)
    assert fit["intercept"] == pytest.approx(-0.08, abs=0.01)
    assert fit["overhead_from_slope"] == pytest.approx(0.08, abs=0.005)
    assert fit["overhead_from_intercept"] == pytest.approx(0.08, abs=0.01)


def test_fit_distinguishes_the_non_slip_loop_with_no_overhead():
    fit = check.fit_row_model(_rows(overhead=0.0))
    assert fit["slope"] == pytest.approx(check.SHIFT_TIME, abs=0.005)
    assert fit["intercept"] == pytest.approx(0.0, abs=0.01)


def test_per_n_table_shows_the_corrected_step_is_constant_and_the_old_one_is_not():
    table = check.per_n_table(_rows(overhead=0.08))
    assert table.loc[3, "corrected_step"] == pytest.approx(table.loc[10, "corrected_step"], abs=0.005)
    assert table.loc[10, "span_over_n"] > table.loc[3, "span_over_n"] + 0.01


def test_row_durations_pairs_markers_with_highlighted_counts_and_skips_mismatched_runs():
    ev = pd.DataFrame({"onset": [1.0, 1.78, 3.0, 4.2], "trial_type": ["ROWS", "ROWE", "ROWS", "ROWE"]})
    rows = check.row_durations(ev, ["全世界", "集合。"])
    assert rows == [(3, pytest.approx(0.78)), (2, pytest.approx(1.2))]
    assert check.row_durations(ev, ["全世界"]) == []
