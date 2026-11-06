import numpy as np
import pytest

from zuco_llm_align.chineseeeg import EYE_CHANNELS, SFREQ
from zuco_llm_align.chineseeeg_regression import (
    CHAR_LAGS,
    BAND_GRIDS,
    CHAR_WIDTHS,
    EYE_LAGS,
    MODELS,
    band_grids,
    build_run,
    char_values,
    evaluate,
    feature_constants,
    layout,
    lexical_block,
    char_features,
    dense_lags,
    fit,
    highlighted_surprisal,
    lag_matrix,
    load_subtlex,
    load_surprisal,
    pearson,
    run_stats,
    select_alphas,
    span,
    stream_of,
    surprisal_bits,
    total,
    vif,
    with_lexical,
)


def test_load_subtlex_reads_log_counts_from_the_gb18030_file(tmp_path):
    path = tmp_path / "SUBTLEX-CH-CHR"
    body = '"Total character count: 1"\t\t\n"Context number: 1"\t\t\nCharacter\tCHRCount\tlogCHR\n我\t100\t2.0\n的\t10\t1.0\n'
    path.write_bytes(body.replace("\n", "\r\n").encode("gb18030"))
    assert load_subtlex(path) == {"我": 2.0, "的": 1.0}


def test_stream_of_keeps_punctuation_for_context_and_indexes_the_highlighted_characters():
    chars, index = stream_of(["我是，他", "\n好。"])
    assert chars == list("我是，他好。")
    assert index == {(0, 0): 0, (0, 1): 1, (0, 2): 3, (1, 0): 4}


def test_surprisal_bits_is_minus_log2_of_the_model_probability_with_nan_where_undefined():
    vocab = 4

    def logprobs(ids):
        return np.log(np.full((len(ids), vocab), 1 / vocab))

    bits = surprisal_bits([1, 2, 0, 3], logprobs, unk=0, window=8, stride=4)
    assert np.isnan(bits[0]) and np.isnan(bits[2])
    assert bits[1] == pytest.approx(2.0) and bits[3] == pytest.approx(2.0)


def test_surprisal_bits_slides_a_window_and_scores_every_position_once_with_full_context():
    seen = []

    def logprobs(ids):
        seen.append(list(ids))
        out = np.full((len(ids), 3), -np.inf)
        out[:, 1] = 0.0
        return out

    bits = surprisal_bits([1] * 10, logprobs, unk=0, window=4, stride=2)
    assert np.isnan(bits[0]) and np.allclose(bits[1:], 0.0)
    assert all(len(s) <= 4 for s in seen)
    assert seen[1] == [1, 1, 1, 1]


def test_highlighted_surprisal_keys_each_highlighted_character_by_row_and_position():
    vocab = {"我": 1, "是": 2, "，": 3, "他": 4}

    def logprobs(ids):
        return np.log(np.full((len(ids), 8), 1 / 8))

    out = highlighted_surprisal(["我是，他", "\n他"], vocab.get, logprobs, unk=0)
    assert set(out) == {(0, 0), (0, 1), (0, 2), (1, 0)}
    assert np.isnan(out[(0, 0)]) and out[(0, 2)] == pytest.approx(3.0)


def test_highlighted_surprisal_marks_characters_outside_the_vocabulary_unknown():
    out = highlighted_surprisal(["我嫣是"], {"我": 1, "是": 2}.get, lambda ids: np.zeros((len(ids), 3)), unk=0)
    assert np.isnan(out[(0, 1)]) and out[(0, 2)] == 0.0


def test_lag_matrix_places_each_event_value_at_its_onset_plus_lag():
    x = lag_matrix(np.array([2, 5]), np.array([[1.0, 3.0], [2.0, 4.0]]), n=10, lags=np.array([0, 1])).toarray()
    assert x.shape == (10, 4)
    assert x[2, 0] == 1.0 and x[3, 1] == 1.0 and x[5, 0] == 2.0 and x[6, 1] == 2.0
    assert x[2, 2] == 3.0 and x[6, 3] == 4.0
    assert x.sum() == 2 * (1 + 2 + 3 + 4)


def test_lag_matrix_drops_lagged_samples_off_the_recording():
    x = lag_matrix(np.array([0, 9]), np.ones((2, 1)), n=10, lags=np.array([-1, 0, 1])).toarray()
    assert x[:, 0].sum() == 1 and x[:, 1].sum() == 2 and x[:, 2].sum() == 1


def test_dense_lags_shifts_each_channel_with_zeros_at_the_edges():
    s = np.arange(5.0)[:, None]
    x = dense_lags(s, np.array([-1, 0, 1]))
    assert np.allclose(x[:, 1], s[:, 0])
    assert np.allclose(x[:, 0], [1, 2, 3, 4, 0]) and np.allclose(x[:, 2], [0, 0, 1, 2, 3])


def test_char_features_fill_missing_values_with_the_mean_and_flag_them():
    f = char_features(np.array([2.0, np.nan, 4.0]), mean=3.0, sd=1.0)
    assert np.allclose(f, [[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])


def _runs(rng, n_runs=4, n=40, n_noise=20):
    out = []
    for _ in range(n_runs):
        x = rng.standard_normal((n, 1 + n_noise))
        out.append(run_stats(x, x[:, :1] + 0.5 * rng.standard_normal((n, 1))))
    return out


def test_fit_on_summed_run_stats_matches_ridge_on_stacked_data():
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((50, 3)) for _ in range(2)]
    ys = [rng.standard_normal((50, 2)) for _ in range(2)]
    w = fit(total([run_stats(x, y) for x, y in zip(xs, ys)]), np.array([1.0, 2.0, 3.0]))
    X, Y = np.vstack(xs), np.vstack(ys)
    assert np.allclose(w, np.linalg.solve(X.T @ X + np.diag([1.0, 2.0, 3.0]), X.T @ Y))


def test_pearson_from_stats_matches_numpy_on_held_out_predictions():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((80, 3)), rng.standard_normal((80, 2))
    w = rng.standard_normal((3, 2))
    pred = x @ w
    expect = [np.corrcoef(pred[:, c], y[:, c])[0, 1] for c in range(2)]
    assert np.allclose(pearson(w, run_stats(x, y)), expect)


def test_run_stats_of_a_column_subset_equal_stats_of_the_subset():
    rng = np.random.default_rng(2)
    x, y = rng.standard_normal((30, 4)), rng.standard_normal((30, 1))
    sub = run_stats(x, y).take([0, 2])
    ref = run_stats(x[:, [0, 2]], y)
    assert np.allclose(sub.G, ref.G) and np.allclose(sub.B, ref.B) and np.allclose(sub.sx, ref.sx)


def test_select_alphas_penalises_a_noise_band_more_than_the_signal_band():
    rng = np.random.default_rng(3)
    runs = _runs(rng)
    bands = [np.array([0]), np.arange(1, 21)]
    alphas, score = select_alphas(runs, bands, grid=np.array([1e-2, 1e1, 1e5]))
    assert alphas[0] < alphas[1]
    assert score > 0.7


def test_char_lags_span_minus_100_to_800_ms_at_64_hz():
    assert CHAR_LAGS[0] == -6 and CHAR_LAGS[-1] == 51 and len(CHAR_LAGS) == 58


def _design_inputs(n_char=6, n_samples=400):
    rng = np.random.default_rng(4)
    data = rng.standard_normal((128, n_samples))
    onsets = 1.0 + 0.42 * np.arange(n_char)
    values = {band: rng.standard_normal((n_char, width)) for band, width in CHAR_WIDTHS.items()}
    values["N"][:, 0] = 1.0
    return data, onsets, values


def test_build_run_columns_follow_the_layout_and_the_span_is_cropped_to_the_reading():
    data, onsets, values = _design_inputs()
    x, y = build_run(data, onsets, np.array([1.0]), np.array([3.4]), values)
    cols = layout()
    assert x.shape[1] == sum(len(cols[b]) for b in CHAR_WIDTHS)
    assert y.shape == (x.shape[0], 128 - len(EYE_CHANNELS))
    first = round(1.0 * SFREQ) + CHAR_LAGS[0]
    last = round(3.4 * SFREQ) + CHAR_LAGS[-1]
    assert x.shape[0] == last - first + 1
    intercept = x[:, cols["N"][:len(CHAR_LAGS)]].toarray()
    assert intercept[0, 0] == 1.0 and intercept.sum() == len(onsets) * len(CHAR_LAGS)


def test_build_run_robust_z_scores_every_target_channel():
    data, onsets, values = _design_inputs()
    _, y = build_run(data * 50 + 7, onsets, np.array([1.0]), np.array([3.4]), values)
    assert np.allclose(np.median(y, axis=0), 0.0, atol=1e-6)


def test_layout_keeps_stroke_columns_inside_the_nuisance_band():
    cols = layout()
    assert set(cols["strokes"]) <= set(cols["C"]) and not set(cols["strokes"]) & set(cols["C-strokes"])
    assert not set(cols["L"]) & set(cols["N"]) and not set(cols["E"]) & set(cols["L"])


def _stats_by_run(rng, n_runs=5):
    cols = layout()
    n_cols = sum(len(cols[b]) for b in CHAR_WIDTHS)
    out = {}
    for rid in range(n_runs):
        x = rng.standard_normal((300, n_cols)) * 0.1
        y = 10 * x[:, cols["L"][:3]].sum(axis=1, keepdims=True) + rng.standard_normal((300, 2))
        out[rid] = run_stats(x, y)
    return out


def test_evaluate_refuses_a_run_on_both_sides_of_the_split():
    stats = _stats_by_run(np.random.default_rng(5))
    with pytest.raises(ValueError, match="both"):
        evaluate(stats, {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}, train=[0, 1], test=[1], grid=np.array([1.0]))


def test_evaluate_scores_every_model_on_the_held_out_runs_and_lexical_signal_shows():
    stats = _stats_by_run(np.random.default_rng(6))
    rows = evaluate(stats, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}, train=[0, 1], test=[2], grid=np.array([1e1, 1e4]))
    by = {r["model"]: r for r in rows}
    assert set(by) == set(MODELS)
    assert by["N+L"]["r"] > by["N"]["r"]
    assert by["N"]["test_runs"] == "4"


def _tables():
    return {"freq": {"我": 3.0, "是": 1.0}, "strokes": {"我": 7, "是": 9, "嫣": -1},
            "surprisal": {(1, 0, 0): np.nan, (1, 0, 1): 4.0, (1, 0, 2): 2.0},
            "vectors": {"我": np.array([1.0, 0.0, 0.0]), "是": np.array([0.0, 1.0, 0.0]), "嫣": np.array([0.0, 0.0, 1.0])}}


def test_feature_constants_come_from_the_given_characters_only():
    consts = feature_constants(list("我是嫣"), [0, 0, 0], [0, 1, 2], [1, 1, 1], _tables(), n_pcs=2)
    assert consts["position"] == pytest.approx((1.0, np.std([0, 1, 2])))
    assert consts["freq"] == pytest.approx((2.0, 1.0))
    assert consts["strokes"] == pytest.approx((8.0, 1.0))
    assert consts["pca"].n_components_ == 2


def test_char_values_build_each_band_with_flags_for_missing_values():
    tables = _tables()
    consts = feature_constants(list("我是嫣"), [0, 0, 0], [0, 1, 2], [1, 1, 1], tables, n_pcs=2)
    v = char_values(list("我是嫣"), [0, 0, 0], [0, 1, 2], 1, tables, consts)
    assert v["N"].shape == (3, CHAR_WIDTHS["N"]) and v["L"].shape == (3, 2) and v["E"].shape == (3, 2)
    assert np.allclose(v["N"][:, 0], 1.0) and np.allclose(v["N"][:, 2], [1, 0, 0])
    assert v["N"][2, 4] == 1.0 and v["N"][2, 3] == 0.0
    assert v["N"][2, 5] == 1.0 and v["L"][2, 0] == 0.0
    assert v["N"][0, 6] == 1.0 and v["L"][1, 1] == pytest.approx(1.0)


def test_vif_is_one_for_independent_columns_and_large_for_near_copies():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2000, 3))
    assert np.allclose(vif(x), 1.0, atol=0.05)
    x[:, 2] = x[:, 0] + 0.01 * rng.standard_normal(2000)
    assert vif(x)[2] > 100


def test_load_surprisal_keys_bits_by_run_row_and_position(tmp_path):
    path = tmp_path / "s.npz"
    np.savez(path, model="m", **{"3_rows": np.array([0, 1]), "3_positions": np.array([2, 0]), "3_bits": np.array([1.5, np.nan])})
    out = load_surprisal(path)
    assert out[(3, 0, 2)] == 1.5 and np.isnan(out[(3, 1, 0)]) and len(out) == 2


def test_lexical_block_with_the_identity_permutation_rebuilds_the_true_n_plus_l_stats():
    data, onsets, values = _design_inputs()
    rs, re_ = np.array([1.0]), np.array([3.4])
    x, y = build_run(data, onsets, rs, re_, values)
    cols = layout()
    nl = np.concatenate([cols["N"], cols["L"]])
    true = run_stats(x, y).take(nl)
    start, _ = span(onsets, rs, re_, data.shape[1])
    block = lexical_block(x[:, cols["N"]], y, np.round(onsets * SFREQ).astype(int) - start, values["L"])
    rebuilt = with_lexical(true, block)
    assert np.allclose(rebuilt.G, true.G) and np.allclose(rebuilt.B, true.B) and np.allclose(rebuilt.sx, true.sx)


def test_with_lexical_swaps_in_permuted_lexical_statistics():
    data, onsets, values = _design_inputs()
    rs, re_ = np.array([1.0]), np.array([3.4])
    x, y = build_run(data, onsets, rs, re_, values)
    cols = layout()
    true = run_stats(x, y).take(np.concatenate([cols["N"], cols["L"]]))
    start, _ = span(onsets, rs, re_, data.shape[1])
    block = lexical_block(x[:, cols["N"]], y, np.round(onsets * SFREQ).astype(int) - start, values["L"][::-1])
    swapped = with_lexical(true, block)
    n = len(cols["N"])
    assert np.allclose(swapped.G[:n, :n], true.G[:n, :n]) and not np.allclose(swapped.B[n:], true.B[n:])


def test_build_run_clips_robust_z_scores_when_asked():
    data, onsets, values = _design_inputs()
    data[5, 150] = 1e6
    _, y = build_run(data, onsets, np.array([1.0]), np.array([3.4]), values, clip=5.0)
    assert np.abs(y).max() == pytest.approx(5.0)
    _, y_raw = build_run(data, onsets, np.array([1.0]), np.array([3.4]), values)
    assert np.abs(y_raw).max() > 5.0


def test_layout_splits_the_nuisance_into_character_and_periocular_bands():
    cols = layout()
    assert len(cols["eye"]) == len(EYE_CHANNELS) * len(EYE_LAGS)
    assert np.array_equal(np.concatenate([cols["C"], cols["eye"]]), cols["N"])
    assert MODELS["N-eye"] == ("C",) and MODELS["N"] == ("C", "eye")


def test_select_alphas_keeps_fixed_leading_penalties_and_searches_the_rest():
    rng = np.random.default_rng(8)
    runs = _runs(rng)
    alphas, _ = select_alphas(runs, [np.array([0]), np.arange(1, 21)], grid=np.array([1e-2, 1e5]), fixed=(1e3,))
    assert alphas[0] == 1e3 and alphas[1] in (1e-2, 1e5)


def test_n_plus_l_plus_e_reuses_the_n_plus_l_penalties():
    stats = _stats_by_run(np.random.default_rng(9))
    rows = {r["model"]: r for r in evaluate(stats, {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}, train=[0, 1], test=[2], grid=np.array([1e1, 1e4]))}
    assert rows["N+L+E"]["alphas"].split()[:3] == rows["N+L"]["alphas"].split()


def test_feature_bands_keep_the_pre_specified_grid_and_nuisance_bands_reach_1e9():
    assert BAND_GRIDS["L"].max() == 1e7 and BAND_GRIDS["E"].max() == 1e7
    assert BAND_GRIDS["eye"].max() == 1e9 and BAND_GRIDS["C"].max() == 1e9
    assert [len(g) for g in band_grids(MODELS["N+L"], BAND_GRIDS)] == [9, 9, 7]


def test_select_alphas_searches_each_band_on_its_own_grid():
    runs = _runs(np.random.default_rng(10))
    alphas, _ = select_alphas(runs, [np.array([0]), np.arange(1, 21)], grid=[np.array([1e-2]), np.array([7.0, 8.0])])
    assert alphas[0] == 1e-2 and alphas[1] in (7.0, 8.0)
