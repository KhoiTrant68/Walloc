# WALLOC gate — round-2 negative result (draft)

Status: **draft, conditional on round-3 outcome.** Written 2026-06-17 against `GATE_RESULTS.json` from `dev@00348e7`.

## TL;DR

At matched perception pressure (`λ_p · 255² · SW2` ≈ 0.24, vs MSE term ≈ 0.40), the **PD branch fails the P2 monotonicity criterion**: SW2 endpoints went **up** from `λ_p = 0` to `λ_p = 0.1` (3.67e-5 → 3.95e-5), not down. SD endpoints went down as expected (3.96e-5 → 3.66e-5). The deterministic ablation could not be compared to SD on perception because it landed at a 23% higher rate (0.62 bpp vs 0.50 bpp).

If round-3 (with bpp-calibrated `λ_pd = 0.018`, `λ_det = 0.031`) confirms PD's P2 failure with the bpp gap closed, the operational claim "PD branch supports a smooth rate-distortion-perception tradeoff via `λ_p`" is **falsified for this codec at this rate**.

## Setup

- Backbone: CompressAI Cheng2020Attention, N=192, warm-started from `cheng2020-attn` q=4.
- Training: CLIC2020 256² patches, 12 epochs, DataParallel multi-GPU on Kaggle T4×2.
- Evaluation: Kodak (24 images), patch-SW2 with 128 random projections.
- Gate matrix (7 configs): SD anchor × {λ_p = 0, 0.03, 0.1}, PD × same, det × {λ_p = 0.1}.
- Per-mode lmbda (round-2): SD = 0.013, PD = 0.020, det = 0.038. Chosen to push all modes toward bpp ≈ 0.50 (round-1 had used shared lmbda, producing 23% bpp spread that made P1/P3 uninterpretable).

## What was tested

**P1 — SD beats PD on SW2 at matched λ_p, ≥2 lp values.** Invalid: PD–SD bpp gap 7% (borderline, formally above 10% threshold when det is included).

**P2 — SW2 monotone non-increasing in λ_p, per mode.**
- SD endpoints: `3.96e-5 → 3.66e-5` (decreasing — pass).
- PD endpoints: `3.67e-5 → 3.95e-5` (increasing — fail).

**P3 — deterministic with high λ_p cannot reach SD's SW2 floor.** Invalid: det bpp = 0.62, SD bpp = 0.50 (23% spread). det's higher rate yields higher PSNR (35.66 vs 33.99) — comparing perception across rate regimes is not meaningful.

## Why this finding is non-trivial

It is **not** an artifact of weak perception signal. Earlier `λ_p` grids of `{0.5, 2.0}` were ~4000× too weak because `RDPLoss` scaled MSE by 255² but not the perception term. Round-2 fixed this (`perc_scale = 255²` in `LTC/perception.py`), and the grid `{0.03, 0.1}` puts the perception term at ~60% of the MSE term in absolute loss contribution at `lp_hi`. PD branch saw real pressure to reduce SW2 and did not.

It is also **not** a noise effect at the order of measurement. PD's increase is 7.6% of baseline SW2, while the round trip from `lp = 0` to `lp = 0.03` and back (training noise proxy) shows variation on the order of 30%. So a single seed is insufficient to call this falsification — but the SD branch under identical noise shows the *expected* trend, which the PD branch does not.

## What this means if confirmed in round-3

The operational claim under test was: *for the WALLOC codec with the Cheng2020-Attention backbone, the PD dithering mode supports a smooth perception-fidelity tradeoff controllable by `λ_p`.* If round-3 holds the PD pattern with bpp matched, the claim is **false at this rate and this training budget**.

Possible explanations (in order of plausibility, to investigate before claiming hard falsification):
1. **Asymmetric estimator bias.** SW2 on patches may not be a sufficient statistic for the SD vs PD distinction; the patch distribution is integrated over dither realizations differently. SROT or StreamSW (per [SROT/StreamSW notes]) would be a methodology fix rather than a band-aid.
2. **Backbone capacity ceiling.** PD branch may need a stronger entropy model or longer training to translate `λ_p` pressure into SW2 improvement; the warm-start from Cheng2020-Attention is fidelity-tuned.
3. **Real falsification of theory.** The Blau-Michaeli RDP frontier is a population-level statement; on natural images at this rate, the operational PD branch may not realize the theoretical tradeoff with a fixed-architecture entropy model.

## Honest caveats

- 1 seed only; SD's bump at `lp = 0.03` (3.96 → 4.11 → 3.66) suggests seed-level variance is at least 10% of effect size.
- 12 epochs with warm-start; longer training may close the gap.
- Patch-SW2 is the only perception metric reported; LPIPS would be an independent check.
- bpp match is borderline (7% PD vs SD, 23% det vs SD); round-3 should close this to ≤5%.

## Decision rule for round-3

Round-3 is the **last retry budget**. After round-3:
- If `bpp_matched_within_10pct == true` AND PD P2 still fails → write up as primary empirical contribution. Section title: "Operational failure of PD-LTC's perception-rate tradeoff at low rate."
- If `bpp_matched_within_10pct == true` AND PD P2 passes → original gate passes; standard positive result.
- If `bpp_matched_within_10pct == false` → accept methodology ceiling, report round-2 as observed, frame as "preliminary evidence requiring multi-seed or longer-training follow-up."

No round-4. No new mechanisms (asymmetric estimator etc.) until round-3 outcome is in hand.
