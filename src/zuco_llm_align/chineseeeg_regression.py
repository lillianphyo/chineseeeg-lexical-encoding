"""Overlap-corrected linear encoding model of continuous ChineseEEG (ds004952) reading EEG.

Plan: lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md. Every character onset gets a response
kernel over CHAR_LAGS, fitted jointly over a run, so the overlapping responses of neighbouring characters are
separated. Feature spaces are bands with their own ridge penalty. A run is summarised by its sufficient
statistics (XᵀX, XᵀY, sums), so fits and held-out Pearson r for any band subset and penalty need no raw data.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from zuco_llm_align.chineseeeg import EYE_CHANNELS, PUNCTUATION, SFREQ

CHAR_LAGS = np.arange(round(-0.1 * SFREQ), round(0.8 * SFREQ) + 1)
EYE_LAGS = np.arange(round(-0.1 * SFREQ), round(0.1 * SFREQ) + 1)
GRID = np.logspace(1, 7, 7)
NUISANCE_GRID = np.logspace(1, 9, 9)


def load_subtlex(path) -> dict[str, float]:
    """Character -> log10 count from the SUBTLEX-CH-CHR file (GB18030, two header lines)."""
    t = pd.read_csv(path, sep="\t", skiprows=2, encoding="gb18030")
    return dict(zip(t.Character.astype(str), t.logCHR.astype(float)))


def load_surprisal(path) -> dict[tuple[int, int, int], float]:
    """(run id, row, position) -> surprisal bits, from `scripts/chineseeeg_surprisal.py`."""
    t = np.load(path)
    out = {}
    for key in t.files:
        if key.endswith("_bits"):
            rid = int(key.split("_")[0])
            out.update({(rid, int(r), int(k)): float(b) for r, k, b in zip(t[f"{rid}_rows"], t[f"{rid}_positions"], t[key])})
    return out


def stream_of(units: list[str]) -> tuple[list[str], dict[tuple[int, int], int]]:
    """The run's text as the LM reads it (line breaks dropped) and (row, position) of each highlighted character -> index."""
    chars, index = [], {}
    for row, unit in enumerate(units):
        k = 0
        for c in unit:
            if c == "\n":
                continue
            if c not in PUNCTUATION:
                index[(row, k)] = len(chars)
                k += 1
            chars.append(c)
    return chars, index


def surprisal_bits(ids, logprobs, unk: int, window: int = 512, stride: int = 256) -> np.ndarray:
    """-log2 p(token | preceding tokens) per position; NaN at position 0 and for unknown tokens.

    `logprobs(ids)` returns (len, vocab) next-token log-probabilities. Windows of `window` tokens step by `stride`;
    each position is scored once, by the first window that covers it after the positions already scored.
    """
    ids = np.asarray(ids)
    bits = np.full(len(ids), np.nan)
    begin, done = 0, 1
    while done < len(ids):
        end = min(begin + window, len(ids))
        lp = logprobs(ids[begin:end])
        for p in range(max(done, begin + 1), end):
            bits[p] = -lp[p - begin - 1, ids[p]] / np.log(2)
        done = end
        begin += stride
    bits[ids == unk] = np.nan
    return bits


def highlighted_surprisal(units: list[str], token_id, logprobs, unk: int, **window) -> dict[tuple[int, int], float]:
    """(row, position) -> surprisal in bits of each highlighted character, given the run's preceding text."""
    chars, index = stream_of(units)
    ids = [token_id(c) for c in chars]
    bits = surprisal_bits([unk if i is None else i for i in ids], logprobs, unk, **window)
    return {key: float(bits[i]) for key, i in index.items()}


def lag_matrix(onsets: np.ndarray, values: np.ndarray, n: int, lags: np.ndarray) -> sparse.csr_matrix:
    """(n, features x lags) design: event value at sample onset + lag, columns feature-major; off-recording samples dropped."""
    n_ev, n_feat = values.shape
    rows = (onsets[:, None, None] + lags[None, None, :]).repeat(n_feat, axis=1)
    cols = np.broadcast_to(np.arange(n_feat)[None, :, None] * len(lags) + np.arange(len(lags)), rows.shape)
    vals = np.broadcast_to(values[:, :, None], rows.shape)
    keep = (rows >= 0) & (rows < n)
    return sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n_feat * len(lags)))


def dense_lags(signal: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """(n, channels x lags): column (c, lag) at sample t holds signal[t - lag, c], zero off the edges."""
    n, ch = signal.shape
    out = np.zeros((n, ch, len(lags)), dtype=signal.dtype)
    for j, lag in enumerate(lags):
        if lag >= 0:
            out[lag:, :, j] = signal[:n - lag]
        else:
            out[:lag, :, j] = signal[-lag:]
    return out.reshape(n, ch * len(lags))


def char_features(values: np.ndarray, mean: float, sd: float) -> np.ndarray:
    """(n, 2): z-scored value with missing set to 0 (the mean), and a missing flag."""
    missing = np.isnan(values)
    return np.column_stack([np.where(missing, 0.0, (values - mean) / sd), missing.astype(float)])


@dataclass
class RunStats:
    G: np.ndarray
    B: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    syy: np.ndarray
    n: int

    def __add__(self, o: "RunStats") -> "RunStats":
        return RunStats(self.G + o.G, self.B + o.B, self.sx + o.sx, self.sy + o.sy, self.syy + o.syy, self.n + o.n)

    def __sub__(self, o: "RunStats") -> "RunStats":
        return RunStats(self.G - o.G, self.B - o.B, self.sx - o.sx, self.sy - o.sy, self.syy - o.syy, self.n - o.n)

    def take(self, cols) -> "RunStats":
        cols = np.asarray(cols)
        return RunStats(self.G[np.ix_(cols, cols)], self.B[cols], self.sx[cols], self.sy, self.syy, self.n)


def run_stats(x, y: np.ndarray) -> RunStats:
    g = x.T @ x
    return RunStats(np.asarray(g.toarray() if sparse.issparse(g) else g, dtype=np.float64), np.asarray(x.T @ y, dtype=np.float64),
                    np.asarray(x.sum(axis=0), dtype=np.float64).ravel(), y.sum(axis=0).astype(np.float64),
                    (y.astype(np.float64) ** 2).sum(axis=0), len(y))


def total(stats: list[RunStats]) -> RunStats:
    out = stats[0]
    for s in stats[1:]:
        out = out + s
    return out


def fit(stats: RunStats, penalty: np.ndarray) -> np.ndarray:
    """Ridge weights (columns, channels) with one penalty per column."""
    return np.linalg.solve(stats.G + np.diag(penalty), stats.B)


def pearson(w: np.ndarray, stats: RunStats) -> np.ndarray:
    """Per-channel Pearson r between X @ w and Y over the samples `stats` summarises."""
    sp = stats.sx @ w
    spp = np.einsum("ic,ij,jc->c", w, stats.G, w)
    spy = (w * stats.B).sum(axis=0)
    n = stats.n
    return (n * spy - sp * stats.sy) / np.sqrt((n * spp - sp ** 2) * (n * stats.syy - stats.sy ** 2))


def penalty_of(bands: list[np.ndarray], alphas) -> np.ndarray:
    p = np.empty(sum(len(b) for b in bands))
    for b, a in zip(bands, alphas):
        p[b] = a
    return p


def select_alphas(groups: list[RunStats], bands: list[np.ndarray], grid, fixed=()) -> tuple[tuple, float]:
    """Per-band penalties maximising the channel-mean r of leave-one-group-out fits within `groups` only.

    `grid` is one array for every band or a list with one array per band. The first len(`fixed`) bands keep the
    given penalties; only the remaining bands are searched.
    """
    whole = total(groups)
    grids = list(grid) if isinstance(grid, list) else [grid] * len(bands)
    best, best_score = None, -np.inf
    for searched in itertools.product(*grids[len(fixed):]):
        alphas = (*fixed, *searched)
        p = penalty_of(bands, alphas)
        score = np.mean([np.nanmean(pearson(fit(whole - g, p), g)) for g in groups])
        if score > best_score:
            best, best_score = alphas, score
    return tuple(float(a) for a in best), float(best_score)


CHAR_WIDTHS = {"N": 7, "L": 2, "E": 10}
STROKE_FEATURES = (3, 4)
MODELS = {"N": ("C", "eye"), "N+L": ("C", "eye", "L"), "N-strokes": ("C-strokes", "eye"), "N+L+E": ("C", "eye", "L", "E"),
          "N-eye": ("C",)}
BASE = {"N+L+E": "N+L"}
BAND_GRIDS = {"C": NUISANCE_GRID, "C-strokes": NUISANCE_GRID, "eye": NUISANCE_GRID, "L": GRID, "E": GRID}


def band_grids(band_names, grids) -> list[np.ndarray]:
    """One penalty grid per band: `grids` maps band -> grid, or is one array for all bands."""
    return [grids[b] if isinstance(grids, dict) else grids for b in band_names]


def layout() -> dict[str, np.ndarray]:
    """Column indices of each band; N = C (character and row events) + eye (periocular regressors). Columns: character nuisance (onset, position, first-in-row, strokes, strokes
    missing, frequency missing, surprisal missing) x lags, row start, row end, periocular channels x EYE_LAGS,
    lexical (log frequency, surprisal) x lags, embedding PCs x lags."""
    n_lag = len(CHAR_LAGS)
    widths = [CHAR_WIDTHS["N"] * n_lag, n_lag, n_lag, len(EYE_CHANNELS) * len(EYE_LAGS), CHAR_WIDTHS["L"] * n_lag, CHAR_WIDTHS["E"] * n_lag]
    edges = np.cumsum([0, *widths])
    cols = {"N": np.arange(edges[0], edges[4]), "L": np.arange(edges[4], edges[5]), "E": np.arange(edges[5], edges[6]),
            "strokes": np.arange(STROKE_FEATURES[0] * n_lag, (STROKE_FEATURES[-1] + 1) * n_lag)}
    cols["C"] = np.arange(edges[0], edges[3])
    cols["eye"] = np.arange(edges[3], edges[4])
    cols["C-strokes"] = np.setdiff1d(cols["C"], cols["strokes"])
    return cols


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0)
    return (x - med) / (1.4826 * np.median(np.abs(x - med), axis=0))


def span(onsets_s, row_starts, row_ends, n_samples: int) -> tuple[int, int]:
    """First and one-past-last sample of the reading span: first event + first lag to last event + last lag."""
    on, rs, re_ = (np.round(np.asarray(t) * SFREQ).astype(int) for t in (onsets_s, row_starts, row_ends))
    start = min(on.min(), rs.min()) + CHAR_LAGS[0]
    if start < 0:
        raise ValueError("reading span starts before the recording")
    return int(start), int(min(max(on.max(), re_.max()) + CHAR_LAGS[-1] + 1, n_samples))


def build_run(data: np.ndarray, onsets_s: np.ndarray, row_starts: np.ndarray, row_ends: np.ndarray,
              values: dict[str, np.ndarray], clip: float | None = None) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Design X and target Y of one run, cropped to the reading span (first event + first lag to last event + last lag).

    `data` is (128 channels, samples) at SFREQ; periocular channels become nuisance regressors, the rest targets,
    both robust z-scored over the span (clipped at +-`clip` if given). `values` holds the per-character band features (CHAR_WIDTHS).
    """
    on, rs, re_ = (np.round(np.asarray(t) * SFREQ).astype(int) for t in (onsets_s, row_starts, row_ends))
    start, stop = span(onsets_s, row_starts, row_ends, data.shape[1])
    eye_rows = [c - 1 for c in EYE_CHANNELS]
    seg = data[:, start:stop].T
    eye, target = robust_z(seg[:, eye_rows]), robust_z(np.delete(seg, eye_rows, axis=1))
    if clip is not None:
        eye, target = np.clip(eye, -clip, clip), np.clip(target, -clip, clip)
    n = stop - start
    blocks = [lag_matrix(on - start, values["N"], n, CHAR_LAGS),
              lag_matrix(rs - start, np.ones((len(rs), 1)), n, CHAR_LAGS),
              lag_matrix(re_ - start, np.ones((len(re_), 1)), n, CHAR_LAGS),
              sparse.csr_matrix(dense_lags(eye, EYE_LAGS)),
              lag_matrix(on - start, values["L"], n, CHAR_LAGS),
              lag_matrix(on - start, values["E"], n, CHAR_LAGS)]
    return sparse.hstack(blocks, format="csr"), target.astype(np.float32)


def model_index(band_names) -> tuple[np.ndarray, list[np.ndarray]]:
    """Design columns of a model and each band's positions within them."""
    cols = layout()
    idx = np.concatenate([cols[b] for b in band_names])
    edges = np.cumsum([0, *(len(cols[b]) for b in band_names)])
    return idx, [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def evaluate(stats: dict[int, RunStats], group_of: dict[int, int], train, test, grid=BAND_GRIDS) -> list[dict]:
    """Fit every model on the train groups (penalties chosen by leave-one-group-out within them) and score the test runs.

    A model in BASE keeps its base model's penalties and searches only its extra band."""
    if set(train) & set(test):
        raise ValueError(f"groups on both sides of the split: {sorted(set(train) & set(test))}")
    train_runs = sorted(r for r in stats if group_of[r] in train)
    test_runs = sorted(r for r in stats if group_of[r] in test)
    if set(train_runs) & set(test_runs):
        raise ValueError("a run is on both sides of the split")
    groups = [total([stats[r] for r in train_runs if group_of[r] == g]) for g in train]
    held = total([stats[r] for r in test_runs])
    rows, chosen = [], {}
    for name, band_names in MODELS.items():
        idx, local = model_index(band_names)
        fixed = chosen[BASE[name]] if name in BASE else ()
        alphas, inner = select_alphas([g.take(idx) for g in groups], local, band_grids(band_names, grid), fixed)
        chosen[name] = alphas
        w = fit(total(groups).take(idx), penalty_of(local, alphas))
        r = pearson(w, held.take(idx))
        rows.append({"model": name, "r": float(np.nanmean(r)), "inner_r": inner, "alphas": " ".join(f"{a:g}" for a in alphas),
                     "train_runs": ",".join(map(str, train_runs)), "test_runs": ",".join(map(str, test_runs))})
    return rows


def _raw_features(chars, rows, ks, rids, tables) -> dict[str, np.ndarray]:
    strokes = np.array([tables["strokes"].get(c, -1) for c in chars], dtype=float)
    return {"position": np.asarray(ks, dtype=float),
            "strokes": np.where(strokes < 0, np.nan, strokes),
            "freq": np.array([tables["freq"].get(c, np.nan) for c in chars]),
            "surprisal": np.array([tables["surprisal"].get((rid, row, k), np.nan) for rid, row, k in zip(rids, rows, ks)])}


def feature_constants(chars, rows, ks, rids, tables, n_pcs: int = CHAR_WIDTHS["E"]) -> dict:
    """Mean and SD of each scalar feature, and the embedding PCA, from the given (training-run) characters only."""
    raw = _raw_features(chars, rows, ks, [rids] * len(chars) if np.isscalar(rids) else rids, tables)
    out = {name: (float(np.nanmean(v)), float(np.nanstd(v))) for name, v in raw.items()}
    out["pca"] = PCA(n_components=n_pcs).fit(np.stack([tables["vectors"][c] for c in chars if c in tables["vectors"]]))
    return out


def char_values(chars, rows, ks, rid: int, tables, consts) -> dict[str, np.ndarray]:
    """Per-character band features of one run (CHAR_WIDTHS order; see `layout`)."""
    raw = _raw_features(chars, rows, ks, [rid] * len(chars), tables)
    z = {name: char_features(v, *consts[name]) for name, v in raw.items()}
    first = (np.asarray(ks) == 0).astype(float)
    n_pcs = consts["pca"].n_components_
    emb = np.zeros((len(chars), n_pcs))
    have = [i for i, c in enumerate(chars) if c in tables["vectors"]]
    if have:
        emb[have] = consts["pca"].transform(np.stack([tables["vectors"][chars[i]] for i in have]))
    return {"N": np.column_stack([np.ones(len(chars)), z["position"][:, 0], first, z["strokes"], z["freq"][:, 1], z["surprisal"][:, 1]]),
            "L": np.column_stack([z["freq"][:, 0], z["surprisal"][:, 0]]),
            "E": emb}


def vif(x: np.ndarray) -> np.ndarray:
    """Variance inflation factor of each column: diagonal of the inverse correlation matrix."""
    return np.diag(np.linalg.inv(np.corrcoef(x, rowvar=False)))


def lexical_block(x_nuisance: sparse.csr_matrix, y: np.ndarray, char_samples: np.ndarray, lexical: np.ndarray) -> dict:
    """Statistics that involve the lexical columns, for lexical values given per character (e.g. permuted)."""
    xl = lag_matrix(char_samples, lexical, x_nuisance.shape[0], CHAR_LAGS)
    return {"LN": np.asarray((xl.T @ x_nuisance).toarray()), "LL": np.asarray((xl.T @ xl).toarray()),
            "B": np.asarray(xl.T @ y, dtype=np.float64), "sx": np.asarray(xl.sum(axis=0)).ravel()}


def add_blocks(a: dict | None, b: dict) -> dict:
    return dict(b) if a is None else {k: a[k] + b[k] for k in a}


def with_lexical(nl: RunStats, block: dict) -> RunStats:
    """N+L statistics (N columns first) with the lexical rows and columns replaced by `block`."""
    n = nl.G.shape[0] - block["LL"].shape[0]
    g = nl.G.copy()
    g[n:, :n], g[:n, n:], g[n:, n:] = block["LN"], block["LN"].T, block["LL"]
    return RunStats(g, np.vstack([nl.B[:n], block["B"]]), np.concatenate([nl.sx[:n], block["sx"]]), nl.sy, nl.syy, nl.n)
