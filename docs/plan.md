# ChineseEEG overlap-corrected regression: lexical features in continuous reading EEG

Status: CODE READY 2026-10-29 (Gate 0 passed, smoke passed). No sulab run yet; launch = Khin. Method #1 of `notes/lit_infotheory_signalproc_reading_eeg_2026-10-29.md`.

## Hypothesis
In held-out ChineseEEG runs, adding a lexical feature space (character log frequency + character surprisal) to an overlap-corrected regression model of the continuous EEG raises prediction accuracy above a nuisance model (character onset, row position, row start/end, stroke count, eye-electrode signals), averaged over the 7 Gate-1 subjects.

## Background
- The contrastive retrieval test found the raw EEG decoder at chance on the locked runs (0.5008). Only a tiny added-value effect survived (+0.009 percentile; `exp-chineseeeg-char-decoding-2026-10.md`). Character identity is a high-entropy target, so a null there does not show the pipeline can detect any known effect.
- Frequency and surprisal effects (N400) are the best-established single-trial lexical effects in reading EEG: Frank et al. 2015 (doi:10.1016/j.bandl.2014.10.006), and Salicchi & Hsu 2025 for Mandarin (COLING 2025.coling-main.246). They act as the positive control the pipeline lacks.
- At 0.42 s/char the responses to neighbouring characters overlap. Regression-based deconvolution fits all event kernels jointly and separates them: Smith & Kutas 2015 (doi:10.1111/psyp.12320), Ehinger & Dimigen 2019 Unfold (doi:10.7717/peerj.7838), Dimigen & Ehinger 2021 (doi:10.1167/jov.21.1.3). Banded ridge gives each feature space its own penalty: Dupré la Tour et al. 2022 (doi:10.1016/j.neuroimage.2022.119728), code `himalaya`.
- Gap (literature check 2026-10-29): no published ChineseEEG analysis tests character-level features against overlap, position and eye confounds.
- Delta vs prior work: Chinese reading FRP regression exists with eye-tracking at sentence scale (Li et al. 2024, Psychophysiology 61(5):e14524, DOI to look up). This is the first test on ChineseEEG's novel-length reading with interpolated onsets and a held-out run split.

## Dataset
- ChineseEEG, OpenNeuro ds004952 v1.2.2. Authors' preprocessed continuous EEG `derivatives/preproc/filtered_0.5_30` (0.5–30 Hz, 128-ch EGI), resampled to 64 Hz. Same loader as the decoding plan (`chineseeeg.py`, `read_raw` with the header fix).
- Subjects: the 7 that passed Gate 1 (sub-04, 06, 07, 08, 09, 10, 14). Rule fixed before decoding; unchanged.
- Runs: GarnettDream 1–18, LittlePrince 2–7 (LP run 1 dropped). Character onsets at each row's own pace (memory `chineseeeg_character_timing`); punctuation is never highlighted, so it has no onset event; it stays in the LM context for surprisal.
- Paths: Mac `~/datasets/ds004952` (sub-04, 05 only → smoke), sulab `F:/datasets/ds004952`. Char table `F:/datasets/ds004952_chartable`.
- New inputs (Gate 0 below): character frequency from SUBTLEX-CH (Cai & Brysbaert 2010, PLoS ONE, doi:10.1371/journal.pone.0010729; verify download and licence) and character surprisal from a character-level Chinese causal LM (candidate `uer/gpt2-chinese-cluecorpussmall`; context = preceding text of the same run, up to 512 characters; fixed before any EEG is fitted).

## Method
- Model: linear encoding model of the continuous EEG (forward model). Time-expanded design (FIR), lags −0.1 to 0.8 s (58 lags at 64 Hz), fitted jointly over all events of a run, so overlapping responses are separated.
- Events and feature spaces (each a band in banded ridge):
  - Nuisance N: character-onset intercept; position in row (numeric) and first-in-row flag; row-start and row-end events; stroke count (visual); the 11 periocular channels (E1, 8, 14, 17, 21, 25, 32, 125–128) as continuous regressors, lags −0.1 to 0.1 s.
  - Lexical L (primary): log SUBTLEX-CH frequency, character surprisal. Both z-scored within subject on training runs.
  - Embedding E (secondary, exploratory): first 10 PCs of the bert-base-chinese static vectors (PCA fit on training-run characters only).
- Neighbours need no separate regressor: character k+1's own features sit on their own lags, so the joint fit already models them.
- Targets: the 117 non-periocular channels.
- Fit: `himalaya` banded ridge, one alpha per band, chosen by inner CV grouped by run within the training runs only. Unknown-frequency characters: value = training mean plus a missing flag in N.
- Pre-fit check: design condition number per subject. The intercept kernel is near-collinear at the fixed pace (expected, not interpreted); feature kernels should be well posed. Report it.

## Baselines
- Nuisance-only model N (the comparator for the primary test).
- Shuffle control: lexical values permuted across characters within each run (onsets kept), 200 permutations per subject → per-subject null for ΔR.
- Earlier result for context: contrastive retrieval at chance (KG 2026-10-29).

## Evaluation
- Split by run. Dev: 4-fold CV over train + val runs (fold = run-order group i mod 5 ∈ {0,1,2,3}). Confirmatory: fit on train + val runs, predict the locked-test runs (i mod 5 = 4: GD5, 10, 15, LP3) once.
- Locked runs note: their EEG was read once on 2026-10-29 for the retrieval H1/H2 (a different model and hypothesis). No regression model has touched them. This plan is committed before any regression code exists. Report this in any paper.
- Harness asserts train ∩ test = ∅ at run level before fitting.
- Primary metric: ΔR = mean over the 117 target channels of Pearson r(predicted, held-out EEG) for N+L minus N. Per subject.
- **H1:** ΔR(L) > 0. One-sided Wilcoxon over 7 subjects, α = 0.05, on the locked runs.
- Secondary, no claim: ΔR of strokes (N without strokes vs N; visual control, expected positive), ΔR(E) beyond N+L, per-subject permutation p, lexical kernel time course and topography (expect a centro-parietal negativity about 300–500 ms for surprisal), dev-CV results.
- One test, one hypothesis. Embedding results are exploratory; a positive one would need its own plan.

- **AMENDMENT (2026-10-29, side-agent review; committed before any dev or locked regression run, zuco-band-features 0184092 / sulab ce4f6fd):** if the chosen penalty of the lexical band is the grid maximum (10⁷), the band is switched off and ΔR(L) is set to exactly 0 (same for the embedding band). The Wilcoxon test uses zero_method="zsplit", so a zero counts half for and half against and all 7 subjects stay in. Reason: a shrunk band leaves a tiny non-zero ΔR (3e-7 in the smoke run) that would count as a win or loss by chance.

## Dev run 1 result (2026-10-29 18:08 → ~20:35, sulab ce4f6fd; locked runs NOT read)
Mean over 4 dev folds, 7 subjects, one-sided Wilcoxon zsplit: lexical ΔR +0.00000 (5/7, p 0.19); strokes ΔR +0.00002 (6/7, p 0.11); embedding −0.00000 (p 0.41). N-model r 0.22–0.54 in 5 subjects (mostly the periocular regressors predicting the other channels), sub-08 r ≈ 0, sub-14 0.07. The positive control (strokes) is not clear, and gains are about 100× below published lexical TRF gains. Lexical band often switched off by the inner CV.
**Dev iteration 2 (decided 2026-10-30 after seeing dev run 1, before any locked read; Khin: yes):** same H1, same split, same models plus a diagnostic. Changes: (1) EEG band-passed 0.5–8 Hz (zero-phase FIR, MNE default) before resampling to 64 Hz, the usual band for lexical TRF effects (Broderick et al. 2018); (2) robust z-scores clipped at ±5 (artifact control; sub-08); (3) diagnostic model N−eye (nuisance without the periocular regressors) to see whether the character response itself is predictable; (4) per-group run statistics saved so later diagnostics need no reload. (5) after the dev-2 smoke (sub-04, 3 runs: N with periocular regressors r −0.022, N−eye +0.056), the nuisance is split into two ridge bands, C (character and row events) and eye (periocular regressors), as banded ridge intends; N+L+E keeps the N+L penalties and searches only E; (6) penalty grid widened to 10¹–10⁹ so the dense eye band can be switched off; the switched-off rule for L/E now uses 10⁹. (7) side-agent review, before any dev-2 result: one shared grid would move the switched-off rule to 10⁹ while L is already off near 10⁷. So each band gets its own grid: L and E keep 10¹–10⁷ (switched-off rule stays at 10⁷, as first pre-specified), C and eye search 10¹–10⁹. Check from the smoke statistics: at 10⁷ the L weights shrink by about 99 % (E 99.96 %), the eye weights only 63 % (99.4 % at 10⁹). Code zuco-band-features 0e5e97f + per-band grids (see KG). The locked-test settings will be fixed from dev iteration 2 and written here before the locked read.

- **AMENDMENT (2026-10-30, leakage audit; zuco-band-features 7c0657d, Mac only, before any locked regression run):** feature means/SDs and the embedding PCA now come from the text of each split's training groups (locked: groups 0–3), not a fixed groups 0–2. Until then dev folds 0–2 scaled with the test fold's text included — stimulus properties only, no EEG, so not a leak, and the locked split was already clean (0–2 ⊂ 0–3); the change makes the code match the Method text. Dev 2 with the old constants finished 12:05 (sulab 26d7b72; kept at `runs/chineseeeg_regression_dev2_consts012`). Khin: rerun. 7c0657d applied on sulab as e283f41 (278 tests pass); dev 2 RESTARTED 16:31 (sulab reflog 16:30; an earlier note said 12:10, wrong) with the per-split constants into `runs/chineseeeg_regression_dev2` (~2.1 h). The dev-2 result reported here is the restarted run; the old-constants folder is for a consistency check only.

- **Power note (2026-10-30, side-agent review; checked with scipy, before any dev-2 result):** with 7 subjects the one-sided Wilcoxon passes only if at most 2 subjects do not gain (all others positive: 0 off p 0.008, 1 off 0.016, 2 off 0.031, 3 off 0.063, 4 off 0.125; 2 off + 1 small loss 0.063). zsplit, Pratt and dropping zeros give the same p here, so the limit is the sample size, not the zero rule. H1 is kept unchanged. Dev 2 will report how many subjects have the lexical band switched off before any locked decision.

- **Dev 2, old-constants run (consistency check only; read 2026-10-30 ~16:45; locked runs NOT read):** `runs/chineseeeg_regression_dev2_consts012`, sulab 26d7b72, 0.5–8 Hz, clip ±5, 4 dev folds × 7 subjects. Lexical band switched off in 0/28 subject-folds (L penalties 10³–10⁵). ΔR(L), mean over folds: positive in 7/7, mean +0.00004 (range +0.00001 to +0.00012), Wilcoxon p 0.008. Strokes control: 6/7, p 0.023 (dev 1: p 0.11). Embedding: 5/7, p 0.23. N r 0.27–0.46; N−eye r 0.05–0.13 (EEG without periocular regressors). Gains are consistent but tiny in r. The official dev-2 result is the restarted per-split run; compare before any locked decision.

- **Dev 2 RESULT (official, per-split constants; 2026-10-30 16:31→19:39, sulab e283f41; locked runs NOT read):** matches the old-constants run to 5 decimals. Lexical band off in 0/28 subject-folds (L penalty 10⁴ in 11, 10⁵ in 17). ΔR(L), mean over 4 folds: 7/7 positive, mean +0.00005 (+0.00001 to +0.00012; 0.004–0.034 % of r(N)), Wilcoxon p 0.008. Strokes control 6/7, p 0.023. Embedding 5/7, p 0.19. r(N) 0.27–0.46; r(N−eye) 0.05–0.13. Verdict: consistent but tiny; power limit not hit.
- **Locked settings FIXED (2026-10-31, before any locked read):** dev-2 settings (0.5–8 Hz band-pass, robust z clipped at ±5, per-band grids, separate eye band). The worker's locked mode had only `--locked-test` (would have used 0.5–30 Hz); now `--locked-test --high 8 --clip 5` (zuco-band-features 99054f9, sulab 66cef61). H1, the zero rule and the effect-size reporting rule unchanged. Launch: `register_chineseeeg_regression.ps1 -Mode locked` (~3.5 h with 200 perms). Khin decides.

## LOCKED RESULT (2026-10-31 08:47→14:23, sulab 66cef61, flags `--locked-test --high 8 --clip 5`; launched from another session with Khin's approval; read once)
- **H1 PASS.** ΔR(L) > 0 in 7/7 subjects, mean +0.000051, bootstrap 95 % CI [+0.000020, +0.000093], one-sided Wilcoxon p 0.0078 (the minimum for n = 7). No subject had the lexical band switched off.
- Secondary (no claim): strokes 6/7, p 0.016; embedding 5/7, p 0.11 (null).
- Effect-size rule (script `zuco-band-features/scripts/chineseeeg_regression_effect_sizes.py`, 4754ed3; output `runs/chineseeeg_regression_locked/effect_sizes.txt`):
  1. Per subject ΔR: +0.000005 to +0.000163.
  2. 0.001–0.04 % of r(N) (0.29–0.52); 0.006–0.17 % of r(N−eye) (0.04–0.11). Tiny.
  3. Within-run permutation (200): 5/7 at the floor p 0.005 (z 3.6–11.5); sub-04 z 0.5 (p 0.32), sub-06 z 1.4 (p 0.12).
  4. Channels: per-channel refit reproduces the CSV mean exactly. Gain is larger on the posterior quarter (group +0.000062) than the frontal quarter (+0.000011); posterior > frontal in 5/7 (sub-06, sub-09 frontal). Not an eye-adjacent effect → "Iterate" condition not met.
  5. Kernels: surprisal GFP peak 312 ms; posterior mean 300–500 ms negative in 6/7 (sub-08 ≈ 0) — the N400 direction (more surprising → more negative). Frequency GFP peak 250 ms, posterior weight at peak positive in 7/7.
- **Verdict:** publishable as a small but consistent positive control (Success criterion → BSPC positive-control section). Wording: "Character surprisal and frequency add a small but consistent amount of held-out variance (7/7 readers; ΔR ≈ 5×10⁻⁵), posterior, with an N400-like surprisal kernel." Never "strong". The locked runs were read once before by the retrieval experiment (state this). Kernel sign is descriptive (no test was pre-specified for it).

- **LOCKED-TEST READ LAUNCHED 2026-10-31 08:47 (Khin: "run the locked test"):** task `chineseeeg_regression_locked`, sulab 66cef61 = Mac 99054f9, `runs/chineseeeg_regression_locked/LOCKED.txt` written at start. Pre-flight: no prior locked folder, no other task running, flags `--locked-test --high 8 --clip 5`. H1 and all rules above fixed before this line. Result read once, via `h1.txt` from the pre-specified `chineseeeg_regression_tests.py`.

## Compute budget
- Gate 0, smoke and code on the Mac (sub-04). Full run on sulab CPU (memory `compute_on_sulab_only`); `himalaya` torch backend on the GPU optional.
- Estimate: design about 2.5M samples × ~600 columns per subject (float32 about 6 GB; accumulate XᵀX per run if memory is short). About 1–2 CPU-h per subject, ~10 h total; well under 24 h.
- Surprisal extraction: once, Mac CPU or sulab, under 1 h.
- smoke: `cd year2/zuco-band-features && python scripts/run_chineseeeg_regression.py --smoke`
- results: year2/zuco-band-features/runs/chineseeeg_regression/results.csv
- No `sulab_cmd:` / `gpu_hours:` / `sulab_results:` lines: the launch stays Khin's decision.

## Gate 0 (before code)
1. SUBTLEX-CH character list downloadable with a usable licence; covers ≥ 95 % of ds004952 character tokens.
2. Chosen LM tokenises characters one-to-one for ≥ 99 % of tokens.
3. Design condition number on sub-04 computed; feature columns not collinear (VIF < 5 between frequency, surprisal, strokes, position).
If 1 or 2 fails: fall back to frequency counted from the ds004952 text (state it), or a different LM; record the change here before fitting.

## Gate 0 result (2026-10-29, before any EEG fit)
1. SUBTLEX-CH (PLOS ONE Files S1, `SUBTLEX-CH-CHR`, "freely available for research purposes"): 99.7 % token / 98.7 % type coverage of the 96,305 highlighted characters. Missing are mostly `*`, `—`, digits. PASS. Mac `~/datasets/subtlex_ch`, sulab `F:/datasets/subtlex_ch`.
2. `uer/gpt2-chinese-cluecorpussmall` (BertTokenizer, one token per character): 0.21 % of tokens unknown (rare characters) → surprisal missing + flag. PASS. Surprisal table: `ds004952_chartable/surprisal.npz` (mean about 4 bits per character).
3. VIF on the 60k dev-training characters: frequency 1.26, surprisal 1.08, strokes 1.19, position 1.00 (frequency–strokes r = −0.40). PASS. Smoke condition number (sub-04, 3 runs): lexical block 2.3 (well posed); the full N+L Gram is singular because the row-start event falls on the same sample as the first-in-row flag and some missing-flag columns are all zero. Ridge handles this; nuisance kernels are not interpreted.
- Implementation notes (no change to the hypothesis): fits use summed per-run statistics (XᵀX, XᵀY) instead of `himalaya`, so 24 runs per subject fit in memory; penalties per band from a grid 10¹–10⁷ by leave-one-group-out within the training groups; feature scaling and embedding PCA from the text of the dev training runs (stimulus only, no EEG); permutation p uses the true fit's penalties.
- Smoke (Mac, sub-04, 4 runs, 46 s): pipeline runs end to end; on 3 training runs the inner CV switched the lexical band off (largest penalty), as expected with so little data. Not a result.
- Code: zuco-band-features 938f224, 8f7c360, 882b77f, c9439f1, d57b053, fbb8386 (sulab fb1e491); 270 tests pass on Mac and sulab. Launch (dev, then locked): `powershell -NoProfile -ExecutionPolicy Bypass -File D:\year2\zuco-llm-alignment\scripts\register_chineseeeg_regression.ps1 -Mode dev` (then `-Mode locked`). CPU only, estimate about 2.5 h dev and 3 h locked.

## Success criterion
- **Publishable (BSPC, as a positive-control section):** H1 passes on the locked runs and the surprisal kernel has the expected sign and latency. Claim: "Continuous reading EEG tracks character frequency/surprisal once overlap, position and eye signals are modelled, while identity retrieval stays at chance." Shows the retrieval null is a power/target problem, not a broken pipeline.
- **Iterate:** H1 passes in dev but not locked, or the effect sits only on frontal/eye-adjacent channels (eye-driven) → consider the fixation-locked reanalysis (shortlist #5).
- **Abandon character-level ChineseEEG work:** H1 fails and strokes ΔR is also ≤ 0 → the interpolated-onset data cannot show even known effects; report as a pitfall with the null-result wording rules.
- **Effect-size reporting rule (2026-10-30, side-agent review of the dev-2 check, before any locked regression read):** dev-2 gains were 7/7 but only +0.00001 to +0.00012 in r (0.01–0.03 % of N's r). With all signs positive the Wilcoxon p (0.008) is the minimum for n = 7 and says nothing about size. If H1 passes, the paper reports, with the p-value and not instead of it: (1) per-subject ΔR(L) and the mean with a bootstrap 95 % CI over subjects; (2) ΔR relative to r(N) and to r(N−eye); (3) per-subject permutation p and z from the 200 within-run permutations (`perms_{s}.csv`); (4) the channels carrying the gain (per-channel ΔR, refit from the saved group sums `stats_{s}.npz` with the penalties in results.csv, no EEG re-read; frontal/periocular-only → "Iterate" above); (5) the surprisal and frequency kernels' sign and latency. The claim is worded as "small but consistent", never as a strong effect. All five come from files the locked run already saves; no code change before the locked read.

## Risks
- Fixed pace → near-collinear intercept kernel; only feature effects are interpreted.
- Interpolated onsets (±tens of ms) blur latency; Gate 1 limits this but does not remove it.
- Eye movements: periocular regressors remove much but not all ocular signal; a frontal-only effect is flagged.
- Surprisal depends on the LM and context length; fixed in Gate 0, not tuned on EEG.
- Frequency and stroke count correlate (frequent characters are simpler); strokes are in N so L must explain variance beyond them.
- Large artifacts remain in some runs (raw SD up to 800 µV): robust z-scoring per run; no run selection on EEG.
- Text overlap: subjects read the same text, but the model is within subject and split by run, so no text crosses train/test within a subject.

## Deliverables
- Code: `year2/zuco-band-features` — `src/zuco_llm_align/chineseeeg_regression.py`, `scripts/run_chineseeeg_regression.py`, tests (≥ 90 % coverage, leakage test).
- Results: `runs/chineseeeg_regression/results.csv` (per subject × feature space × split), kernels `.npz`, figures (ΔR per subject, surprisal kernel topography).
- Commit hash in run snapshot; KG entry under Experiments → Completed.

## Links
- Decoding plan and locked result: `exp-chineseeeg-char-decoding-2026-10.md`
- Literature: `notes/lit_infotheory_signalproc_reading_eeg_2026-10-29.md`, `notes/chineseeeg_literature_timing_2026-10-28.md`
- Memories: `chineseeeg_character_timing`, `neighbour_word_window_control`, `null_result_wording`, `data_leakage_guard`

## AMENDMENT 2026-11-02 — character-onset correction and a second locked read (before the re-run)
- The advisor (2026-10-31) asked for the 0.42 s interval to be verified from the stimulus code and the marker meanings. In `play_novel.py` (eye-tracker loop) each character lasts 0.35 s of display plus ~0.08 s of stimulus rebuild; ROWS is sent at the onset of character 1 and ROWE after the 0.35 s display of character n. Marker regression over 22,298 rows (`scripts/chineseeeg_row_duration_check.py`): duration = 0.4268 n − 0.086 s, as that loop predicts.
- Consequence: the step inside a row is (ROWE − ROWS − 0.35)/(n − 1) for n ≥ 2, not (ROWE − ROWS)/n. The first formula places onsets early by k × ~8 ms (≤ 70 ms at the row end, ~35 ms mean). The row-drop rule is unchanged, so the same rows enter both analyses.
- Decision (Khin, 2026-11-02): re-run the development folds and the locked test with the corrected onsets. This is a second read of the locked runs for this experiment. Both reads are reported; the second is labelled as the pre-specified test repeated after an onset correction found in the advisor's review. H1, the zero rule, the effect-size rule and all settings are unchanged. Outputs: `runs/chineseeeg_regression_dev_corrected` (development repository), `runs/chineseeeg_regression_locked_corrected`; the first read is kept in `runs/locked_first_read_old_onsets`.

## AMENDMENT 2026-11-03 — post hoc checks for the advisor's review (written before their results)
- Development re-run matched the earlier one (lexical gain 7/7, strokes control negative, embedding gain null). Second locked read launched 2026-11-02 23:55.
- Timing gate re-check: the 7 readers were chosen with span / n onsets. The gate is re-run on all 10 readers with the corrected onsets, train and validation runs only. Same 7 pass → one sentence in the paper; a different set → the locked read is repeated for that set.
- Each feature alone (point 4): from the saved training statistics of the second read, one lexical feature at a time with the N+L penalties; no EEG re-read; post hoc, no claim.
- Structure-preserving null and penalty re-selection (point 3): `--null shuffle circular --reselect` (circular shift of the paired lexical sequence by ≥ 100 characters within each run; lexical penalty re-chosen per surrogate by leave-one-group-out, nuisance penalties kept). This re-reads the locked runs only to build surrogates; the observed ΔR is unchanged by construction. Launched only with Khin's approval; reported as post hoc sensitivity analyses with their assumptions.

## RESULT 2026-11-03 — post hoc checks (written after their results)
- Timing gate re-check: same 7 readers; S13 marginal, not added. Paper Section 3.
- Each feature alone: frequency 6/7 (3.2 × 10⁻⁵), surprisal 7/7 (2.8 × 10⁻⁵). Paper Section 5.
- Surrogate pass `runs/chineseeeg_regression_locked_nulls` (05:40–17:07, git e844baf): observed gains identical to the second read in all 7 readers. Circular null with fixed penalties: 4/7 at the floor, S08 0.010; re-selected penalty: null means → 0 (−0.5 to +0.3 × 10⁻⁵), 5/7 at the floor under both nulls; S04 and S06 above 0.05 except S06 circular + re-selection (0.030). Paper Section 5, Table 3. No claim changes.
