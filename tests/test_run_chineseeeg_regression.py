import numpy as np
import pandas as pd
import pytest

from zuco_llm_align.chineseeeg import GATE1_PASS, run_id
from zuco_llm_align.chineseeeg_regression import CHAR_LAGS, CHAR_WIDTHS, layout, run_stats

from scripts.chineseeeg_regression_tests import tests as regression_tests
from scripts import run_chineseeeg_regression as runner
from scripts.run_chineseeeg_regression import GROUP_OF, LOCKED_GROUP, condition, constants, kernels, permutation_deltas


def test_locked_group_is_the_pre_specified_locked_runs():
    locked = {r for r, g in GROUP_OF.items() if g == LOCKED_GROUP}
    assert locked == {run_id("GarnettDream", 5), run_id("GarnettDream", 10), run_id("GarnettDream", 15), run_id("LittlePrince", 3)}


def test_constants_come_from_the_text_of_the_given_groups_only(monkeypatch):
    seen = []

    def units(root, session, run):
        seen.append(GROUP_OF[run_id(session, run)])
        return ["我是" if GROUP_OF[run_id(session, run)] == 0 else "是是"]

    monkeypatch.setattr(runner, "units_of", units)
    tables = {"freq": {"我": 3.0, "是": 1.0}, "strokes": {"我": 7, "是": 9}, "surprisal": {},
              "vectors": {"我": np.array([1.0, 0.0]), "是": np.array([0.0, 1.0])}}
    a = constants("root", tables, (0,), n_pcs=1)
    assert set(seen) == {0}
    b = constants("root", tables, (1,), n_pcs=1)
    assert a["freq"] == pytest.approx((2.0, 1.0)) and b["freq"] == pytest.approx((1.0, 0.0))


def _stats(rng):
    cols = layout()
    n_cols = sum(len(cols[b]) for b in CHAR_WIDTHS)
    out = {}
    for rid in [r for r, g in GROUP_OF.items() if g in (0, 1, 4)][:6]:
        x = rng.standard_normal((200, n_cols)) * 0.1
        out[rid] = run_stats(x, 10 * x[:, cols["L"][:2]].sum(axis=1, keepdims=True) + rng.standard_normal((200, 2)))
    return out


def test_kernels_have_one_time_course_per_lexical_feature_and_channel():
    w, alphas = kernels(_stats(np.random.default_rng(0)), (0, 1), np.array([1e1, 1e4]))
    assert w.shape == (2, len(CHAR_LAGS), 2) and len(alphas) == 3


def test_condition_reports_both_gram_matrices():
    c = condition(_stats(np.random.default_rng(1)), (0, 1))
    assert set(c) == {"N+L", "L"} and c["L"] >= 1.0


def test_permutation_deltas_give_one_delta_per_permutation():
    stats = _stats(np.random.default_rng(2))
    cols = layout()
    n_l = len(cols["L"])
    block = {"LN": np.zeros((n_l, len(cols["N"]))), "LL": np.eye(n_l), "B": np.zeros((n_l, 2)), "sx": np.zeros(n_l)}
    perm = {g: [block, block] for g in (0, 1, 4)}
    d = permutation_deltas(stats, perm, (0, 1), (4,), np.array([1e1, 1e4]))
    assert list(d.columns) == ["delta_r"] and len(d) == 2 and np.allclose(d.delta_r[0], d.delta_r[1])


def test_permutation_deltas_can_reselect_the_lexical_penalty_per_surrogate():
    stats = _stats(np.random.default_rng(3))
    cols = layout()
    n_l = len(cols["L"])
    block = {"LN": np.zeros((n_l, len(cols["N"]))), "LL": np.eye(n_l), "B": np.zeros((n_l, 2)), "sx": np.zeros(n_l)}
    grid = np.array([1e1, 1e4])
    d = permutation_deltas(stats, {g: [block] for g in (0, 1, 4)}, (0, 1), (4,), grid, reselect=True)
    assert list(d.columns) == ["delta_r", "alpha_l", "delta_r_reselected"]
    assert d.alpha_l[0] in grid
    assert np.isfinite(d.delta_r_reselected[0])


def test_surrogate_shuffle_is_a_permutation_and_circular_keeps_the_serial_order():
    rng = np.random.default_rng(0)
    n = 500
    shuffled = runner.surrogate_order(n, "shuffle", rng)
    assert sorted(shuffled) == list(range(n)) and not np.array_equal(shuffled, np.arange(n))
    for _ in range(20):
        rolled = runner.surrogate_order(n, "circular", rng)
        assert sorted(rolled) == list(range(n))
        assert set(np.diff(rolled) % n) == {1}
        shift = int(rolled[0])
        assert runner.MIN_SHIFT <= shift <= n - runner.MIN_SHIFT


def test_surrogate_circular_shift_of_a_short_run_still_moves_every_value():
    rng = np.random.default_rng(0)
    rolled = runner.surrogate_order(6, "circular", rng)
    assert sorted(rolled) == list(range(6)) and rolled[0] != 0


def test_surrogate_order_rejects_an_unknown_null():
    with pytest.raises(ValueError):
        runner.surrogate_order(10, "block", np.random.default_rng(0))


def _results(gain, off=()):
    rows = []
    for i, s in enumerate(GATE1_PASS):
        base = 0.02 + 0.001 * i
        a_l = "1e+07" if s in off else "100"
        alphas = {"N": "10", "N+L": f"10 {a_l}", "N-strokes": "10", "N+L+E": f"10 {a_l} 100"}
        for model, r in {"N": base, "N+L": base + gain, "N-strokes": base - 0.001, "N+L+E": base + gain}.items():
            rows.append({"subject": s, "split": "locked", "model": model, "r": r, "alphas": alphas[model]})
    return pd.DataFrame(rows)


def test_a_switched_off_lexical_band_counts_as_exactly_zero_and_half_a_win():
    out = regression_tests(_results(3e-7, off=GATE1_PASS[:4]))
    assert out["lexical_zero"] == 4 and out["per_subject"].loc["sub-04", "lexical"] == 0.0
    assert not out["h1_pass"]


def test_h1_passes_when_every_subject_gains_and_fails_without_a_gain():
    assert regression_tests(_results(0.002))["h1_pass"]
    out = regression_tests(_results(-0.002))
    assert not out["h1_pass"] and out["strokes_positive"] == len(GATE1_PASS)


def test_tests_refuse_missing_subjects():
    with pytest.raises(ValueError, match="missing"):
        regression_tests(_results(0.002).query("subject != 'sub-04'"))
