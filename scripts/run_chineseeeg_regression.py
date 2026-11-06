"""Overlap-corrected lexical regression on continuous ChineseEEG (ds004952) EEG, one subject per call.

Plan: lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md. Groups = run-order position i mod 5.
Dev: 4-fold CV over groups 0-3; the locked-test runs (group 4) are never read. --locked-test: fit on groups 0-3,
score group 4 once, plus a within-run permutation of the lexical values (penalties fixed from the true fit).
Feature scaling and the embedding PCA come from the text of each split's training groups (no EEG).
Rows for the same subject and split are replaced, so per-subject calls merge into one results.csv.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zuco_llm_align.chineseeeg import (
    GATE1_PASS,
    RUN_ORDER,
    SFREQ,
    char_onsets,
    load_char_table,
    read_raw,
    run_id,
    runs_of,
    units_of,
)
from zuco_llm_align.chineseeeg_regression import (
    BAND_GRIDS,
    MODELS,
    add_blocks,
    band_grids,
    build_run,
    char_values,
    evaluate,
    feature_constants,
    fit,
    layout,
    lexical_block,
    model_index,
    load_subtlex,
    load_surprisal,
    pearson,
    penalty_of,
    run_stats,
    select_alphas,
    span,
    stream_of,
    total,
    with_lexical,
)

from zuco_llm_align.provenance import git_hash

GROUP_OF = {run_id(s, r): i % 5 for i, (s, r) in enumerate(RUN_ORDER)}
DEV_GROUPS = (0, 1, 2, 3)
LOCKED_GROUP = 4


def build_parser() -> argparse.ArgumentParser:
    home = Path.home() / "datasets"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.environ.get("CHINESEEEG_ROOT", str(home / "ds004952")))
    p.add_argument("--table", default=str(home / "ds004952_chartable/char_table.npz"))
    p.add_argument("--surprisal", default=str(home / "ds004952_chartable/surprisal.npz"))
    p.add_argument("--subtlex", default=str(home / "subtlex_ch/SUBTLEX-CH-CHR"))
    p.add_argument("--subject", default=GATE1_PASS[0], choices=GATE1_PASS)
    p.add_argument("--out", default="runs/chineseeeg_regression/results.csv")
    p.add_argument("--locked-test", action="store_true")
    p.add_argument("--perms", type=int, default=200, help="lexical permutations (locked test only)")
    p.add_argument("--null", nargs="+", choices=NULLS, default=["shuffle"],
                   help="surrogate kinds: within-run shuffle (pre-specified) and/or circular shift (post hoc)")
    p.add_argument("--reselect", action="store_true", help="also re-choose the lexical penalty for every surrogate (post hoc)")
    p.add_argument("--high", type=float, default=None, help="low-pass edge in Hz (band-pass from 0.5 Hz) before resampling")
    p.add_argument("--clip", type=float, default=None, help="clip robust z-scores at +-this value")
    p.add_argument("--smoke", action="store_true", help="sub-04, first 4 runs, small grid, 2 permutations")
    return p


def text_of(root, session, run):
    chars, index = stream_of(units_of(root, session, run))
    return [(chars[i], row, k) for (row, k), i in index.items()]


def constants(root, tables, groups, **kw) -> dict:
    """Feature scaling and embedding PCA from the text of the runs in `groups` (the split's training groups), no EEG."""
    chars, rows, ks, rids = [], [], [], []
    for s, r in RUN_ORDER:
        if GROUP_OF[run_id(s, r)] in groups:
            for c, row, k in text_of(root, s, r):
                chars.append(c), rows.append(row), ks.append(k), rids.append(run_id(s, r))
    return feature_constants(chars, rows, ks, rids, tables, **kw)


def load_runs(root, subject, wanted, high=None) -> dict[int, dict]:
    """Preprocessed EEG (µV, float32) and events of every wanted run, read once; designs are built per split."""
    runs = {}
    for session, run, stem in runs_of(Path(root), subject):
        rid = run_id(session, run)
        if rid not in wanted:
            continue
        events = pd.read_csv(f"{stem}_events.tsv", sep="\t")
        onsets, _ = char_onsets(events, units_of(root, session, run))
        raw = read_raw(f"{stem}_eeg.vhdr")
        if high is not None:
            raw.filter(0.5, high, verbose=False)
        raw.resample(SFREQ, verbose=False)
        runs[rid] = {"data": (raw.get_data() * 1e6).astype(np.float32), "t": np.array([o for o, *_ in onsets]),
                     "rs": events.loc[events.trial_type == "ROWS", "onset"].to_numpy(),
                     "re": events.loc[events.trial_type == "ROWE", "onset"].to_numpy(),
                     "chars": [c for _, c, _, _ in onsets], "rows": [row for *_, row, _ in onsets], "ks": [k for *_, k in onsets]}
        print(f"  {subject} {session} run {run}: {len(onsets)} chars, {runs[rid]['data'].shape[1]} samples", flush=True)
    return runs


NULLS = ("shuffle", "circular")
MIN_SHIFT = 100


def surrogate_order(n: int, kind: str, rng) -> np.ndarray:
    """Character order of one surrogate: a within-run shuffle, or a circular shift by at least MIN_SHIFT characters
    (fewer on a run shorter than 2 * MIN_SHIFT), which keeps the serial order of the lexical values."""
    if kind == "shuffle":
        return rng.permutation(n)
    if kind != "circular":
        raise ValueError(f"unknown null {kind!r}; choose from {NULLS}")
    lo = max(1, min(MIN_SHIFT, n // 2))
    return np.roll(np.arange(n), int(rng.integers(lo, max(lo, n - lo) + 1)))


def split_stats(runs, tables, consts, clip, perm_split, n_perms, seed, nulls=("shuffle",)):
    """Run statistics with the split's feature constants, and per null kind the surrogate lexical blocks summed per group."""
    cols = layout()
    rng = np.random.default_rng(seed)
    groups = sorted({GROUP_OF[rid] for rid in runs})
    stats, perm = {}, {kind: {g: [None] * n_perms for g in groups} for kind in nulls}
    for rid, r in runs.items():
        values = char_values(r["chars"], r["rows"], r["ks"], rid, tables, consts)
        x, y = build_run(r["data"], r["t"], r["rs"], r["re"], values, clip=clip)
        stats[rid] = run_stats(x, y)
        if n_perms:
            start, _ = span(r["t"], r["rs"], r["re"], r["data"].shape[1])
            samples = np.round(r["t"] * SFREQ).astype(int) - start
            xn = x[:, cols["N"]]
            for p in range(n_perms):
                for kind in nulls:
                    block = lexical_block(xn, y, samples, values["L"][surrogate_order(len(r["t"]), kind, rng)])
                    perm[kind][GROUP_OF[rid]][p] = add_blocks(perm[kind][GROUP_OF[rid]][p], block)
    return stats, perm


def permutation_deltas(stats, perm, train, test, grid, reselect: bool = False) -> pd.DataFrame:
    """Delta r (N+L - N) on the test runs per surrogate; `perm` maps group -> surrogate lexical blocks.

    `delta_r` keeps the true fit's penalties. With `reselect`, `alpha_l` is the lexical penalty re-chosen for the
    surrogate by leave-one-group-out within the train groups (nuisance penalties kept) and `delta_r_reselected` its gain."""
    n, n_bands = model_index(MODELS["N"])
    nl, bands = model_index(MODELS["N+L"])
    tr = total([stats[r] for r in stats if GROUP_OF[r] in train])
    te = total([stats[r] for r in stats if GROUP_OF[r] in test])
    groups = [total([stats[r] for r in stats if GROUP_OF[r] == g]) for g in train]
    a_n, _ = select_alphas([g.take(n) for g in groups], n_bands, band_grids(MODELS["N"], grid))
    r_n = np.nanmean(pearson(fit(tr.take(n), penalty_of(n_bands, a_n)), te.take(n)))
    a_nl, _ = select_alphas([g.take(nl) for g in groups], bands, band_grids(MODELS["N+L"], grid))
    pen = penalty_of(bands, a_nl)
    rows = []
    for p in range(len(perm[train[0]])):
        bt = total_blocks([perm[g][p] for g in train])
        bs = total_blocks([perm[g][p] for g in test])
        tr_p, te_p = with_lexical(tr.take(nl), bt), with_lexical(te.take(nl), bs)
        row = {"delta_r": float(np.nanmean(pearson(fit(tr_p, pen), te_p)) - r_n)}
        if reselect:
            folds = [with_lexical(g.take(nl), perm[grp][p]) for grp, g in zip(train, groups)]
            a_p, _ = select_alphas(folds, bands, band_grids(MODELS["N+L"], grid), fixed=a_nl[:-1])
            row["alpha_l"] = a_p[-1]
            row["delta_r_reselected"] = float(np.nanmean(pearson(fit(tr_p, penalty_of(bands, a_p)), te_p)) - r_n)
        rows.append(row)
    return pd.DataFrame(rows)


def total_blocks(blocks: list[dict]) -> dict:
    out = None
    for b in blocks:
        out = add_blocks(out, b)
    return out


def kernels(stats, train, grid) -> tuple[np.ndarray, tuple]:
    """Lexical kernels (feature, lag, channel) of the N+L model fitted on the train groups."""
    nl, bands = model_index(MODELS["N+L"])
    groups = [total([stats[r] for r in stats if GROUP_OF[r] == g]) for g in train]
    alphas, _ = select_alphas([g.take(nl) for g in groups], bands, band_grids(MODELS["N+L"], grid))
    w = fit(total(groups).take(nl), penalty_of(bands, alphas))
    return w[bands[-1]].reshape(2, -1, w.shape[1]), alphas


def condition(stats, train) -> dict:
    """Condition numbers of the unpenalised N+L and lexical-only Gram matrices of the train groups (plan Gate 0, check 3)."""
    cols = layout()
    g = total([stats[r] for r in stats if GROUP_OF[r] in train])
    return {"N+L": float(np.linalg.cond(g.take(np.concatenate([cols["N"], cols["L"]])).G)), "L": float(np.linalg.cond(g.take(cols["L"]).G))}


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke and args.locked_test:
        raise SystemExit("--smoke and --locked-test cannot be combined: the locked test is read once")
    subject = GATE1_PASS[0] if args.smoke else args.subject
    vectors, strokes = load_char_table(args.table)
    tables = {"freq": load_subtlex(args.subtlex), "strokes": strokes, "surprisal": load_surprisal(args.surprisal), "vectors": vectors}
    grid = np.logspace(2, 6, 3) if args.smoke else BAND_GRIDS
    if args.smoke:
        wanted = {run_id(s, r) for s, r in RUN_ORDER[:4]}
        splits = {"dev3": ((0, 1, 2), (3,))}
        n_perms = 2
    elif args.locked_test:
        wanted = set(GROUP_OF)
        splits = {"locked": (DEV_GROUPS, (LOCKED_GROUP,))}
        n_perms = args.perms
    else:
        wanted = {r for r, g in GROUP_OF.items() if g in DEV_GROUPS}
        splits = {f"dev{k}": (tuple(g for g in DEV_GROUPS if g != k), (k,)) for k in DEV_GROUPS}
        n_perms = 0
    perm_split = next(iter(splits.values())) if n_perms else None
    runs = load_runs(args.root, subject, wanted, high=args.high)
    n_chars = {rid: len(r["chars"]) for rid, r in runs.items()}

    rows, consts, first = [], {}, None
    for name, (train, test) in splits.items():
        consts[name] = constants(args.root, tables, train)
        split_perms = n_perms if (train, test) == perm_split else 0
        stats, perm = split_stats(runs, tables, consts[name], args.clip, perm_split, split_perms, seed=int(subject[-2:]), nulls=tuple(args.null))
        first = first or (stats, perm)
        for row in evaluate(stats, GROUP_OF, train, test, grid):
            rows.append({"subject": subject, "split": name, **row})
    stats, perm = first
    by = {r["model"]: r["r"] for r in rows if r["split"] == next(iter(splits))}
    print(pd.DataFrame(rows)[["split", "model", "r", "alphas"]].to_string(index=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        new = pd.concat([old[~((old.subject == subject) & old.split.isin(list(splits)))], new], ignore_index=True)
    new.to_csv(out, index=False)

    sums = {g: total([v for r, v in stats.items() if GROUP_OF[r] == g]) for g in sorted({GROUP_OF[r] for r in stats})}
    np.savez_compressed(out.parent / f"stats_{subject}.npz", **{f"g{g}_{k}": np.asarray(getattr(v, k), dtype=np.float32)
                                                                for g, v in sums.items() for k in ("G", "B", "sx", "sy", "syy", "n")})
    train_groups = next(iter(splits.values()))[0]
    w, alphas = kernels(stats, train_groups, grid)
    np.savez(out.parent / f"kernels_{subject}.npz", lexical=w, features=np.array(["log_frequency", "surprisal"]), lags=np.arange(w.shape[1]))
    snapshot = {"subject": subject, "git_hash": git_hash(), "dataset": "ds004952 v1.2.2", "args": vars(args), "grid": {b: g.tolist() for b, g in grid.items()} if isinstance(grid, dict) else grid.tolist(),
                "splits": {k: [list(a), list(b)] for k, (a, b) in splits.items()}, "n_chars": {str(k): v for k, v in n_chars.items()},
                "constants": {s: {k: v for k, v in c.items() if k != "pca"} for s, c in consts.items()}, "kernel_alphas": alphas, "models": list(MODELS),
                "condition": condition(stats, train_groups)}
    if n_perms:
        observed = by["N+L"] - by["N"]
        snapshot["nulls"] = {}
        for kind in args.null:
            deltas = permutation_deltas(stats, perm[kind], *perm_split, grid, reselect=args.reselect)
            p_of = {c: float((1 + (deltas[c] >= observed).sum()) / (1 + n_perms)) for c in deltas.columns if c.startswith("delta_r")}
            snapshot["nulls"][kind] = {"n": n_perms, "observed_delta_r": observed, "p": p_of["delta_r"],
                                       **({"p_reselected": p_of["delta_r_reselected"]} if args.reselect else {})}
            suffix = "" if kind == "shuffle" else f"_{kind}"
            deltas.insert(0, "perm", np.arange(n_perms))
            deltas.insert(0, "subject", subject)
            deltas.to_csv(out.parent / f"perms{suffix}_{subject}.csv", index=False)
            print(f"{kind} null p = {p_of} (observed delta r {observed:.4f})", flush=True)
        snapshot["permutation"] = snapshot["nulls"][args.null[0]]
    (out.parent / f"snapshot_{subject}.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
