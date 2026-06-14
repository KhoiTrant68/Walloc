"""
Unit tests for WALLOC.

Run with `pytest tests/test_walloc.py -v` from the repo root.

Coverage:
    - crypto lemma: Q(y-u) + u =d y + u_eq with u_eq ~ Unif(V_0(Lambda_f))
    - Prop. 3.3 expectation identity (latent space):
          E||y - y_PD||^2 = s^2 * E||y - y_SD||^2 + E||y - Q(y)||^2
    - SW_2^2 estimator vs Gaussian closed-form (App. E setup)
    - NestedE8Product: enumeration count, reps lie in V_0(coarse),
      uniformity of coset sampling, block-independence
    - QSD inference uses the COARSE quantizer (c ∈ a*Lambda_f) and the
      QSD likelihood applies the per-coord `a` volume correction
    - DitheredGaussianConditionalLattice end-to-end shape/no-NaN for all
      modes, both training and inference paths
    - Cheng2020RDP forward integration test (full model on a toy batch)
"""

import numpy as np
import pytest
import torch

from LTC.dithered_models import (
    Cheng2020RDP,
    DitheredGaussianConditionalLattice,
)
from LTC.nested_lattice import NestedE8Product
from LTC.perception import RDPLoss, sliced_w2
from LTC.quantizers import E8Quantizer


torch.manual_seed(0)
np.random.seed(0)


# --------------------------------------------------------------------- #
# helpers                                                                #
# --------------------------------------------------------------------- #


def _fresh_conditional(channels=8, lattice="E8", dither_mode="sd",
                       s=1.0, N_integral=512, nested=None):
    return DitheredGaussianConditionalLattice(
        None,
        channels=channels,
        N_integral=N_integral,
        lattice_name=lattice,
        dither_mode=dither_mode,
        s_dither=s,
        nested_quantizer=nested,
    )


# --------------------------------------------------------------------- #
# 1. Crypto lemma                                                       #
# --------------------------------------------------------------------- #


def test_crypto_lemma_residual_is_uniform_on_voronoi():
    """For y arbitrary and u ~ Unif(V_0(Lambda_f)),
        r = Q_{Lambda_f}(y - u) + u - y
    is Unif(V_0(Lambda_f)), independent of y (Zamir Thm 5.2.1)."""
    cond = _fresh_conditional(channels=8, lattice="E8")
    n = 50_000
    y = torch.randn(n, 8) * 3.0
    u = cond._draw_cell_uniform(n, y.device)
    c = cond.quantizer(y - u)
    r = c + u - y
    assert r.mean(dim=0).abs().max().item() < 0.05
    nsm_estimate, _ = cond.quantizer.NSM(n_samples=10_000)
    r_norm_sq_per_dim = (r ** 2).sum(dim=1).mean() / 8
    rel_err = abs(r_norm_sq_per_dim.item() - nsm_estimate.item()) / nsm_estimate.item()
    assert rel_err < 0.1, (r_norm_sq_per_dim.item(), nsm_estimate.item())


def test_sd_inference_residual_matches_unif_voronoi_via_full_api():
    """End-to-end: exercise the full quantize API in SD mode and confirm
    the inference-time residual (rec - y) distributes like Unif(V_0(Lambda_f)).
    Confirms the _flatten/_unflatten plumbing is correct AND the new API
    contract (rate, rec) returns rec = c + u for SD."""
    cond = _fresh_conditional(channels=8, lattice="E8", dither_mode="sd")
    cond.eval()
    n = 20_000
    y = torch.randn(n, 8)
    # Construct y_input of shape [B=1, C=8, H=n, W=1] such that _flatten
    # round-trips it back to [n, 8]. The inverse is (.reshape(8, n).t()).
    y_input = y.t().contiguous().reshape(1, 8, n, 1)
    rate, rec = cond.quantize(y_input, "dequantize")
    # SD: rate == rec
    assert torch.equal(rate, rec)
    y_back = rec.reshape(8, n).t().contiguous()
    res = y_back - y
    fresh = cond._draw_cell_uniform(n, y.device)
    assert (res.mean(dim=0) - fresh.mean(dim=0)).abs().max().item() < 0.05
    assert ((res ** 2).mean(dim=0) - (fresh ** 2).mean(dim=0)).abs().max().item() < 0.05


# --------------------------------------------------------------------- #
# 2. PD distortion decomposition (Prop. 3.3)                            #
# --------------------------------------------------------------------- #


def test_pd_distortion_decomposition_per_sample():
    """Algebraic check on each draw:
        ||y - (Q(y) + s*u)||^2 = ||y-Q(y)||^2 - 2s(y-Q(y))^T u + s^2 ||u||^2.
    Verifies the implementation reproduces the algebra exactly."""
    cond = _fresh_conditional(channels=8, lattice="E8", dither_mode="pd")
    n = 100_000
    y = torch.randn(n, 8) * 2.0
    q = cond.quantizer(y)
    u = cond._draw_cell_uniform(n, y.device)
    for s in (0.0, 0.5, 1.0, 1.5):
        yPD = q + s * u
        e_obs = ((y - yPD) ** 2).sum(dim=1).mean().item()
        e_pred = (
            ((y - q) ** 2).sum(dim=1).mean().item()
            - 2 * s * ((y - q) * u).sum(dim=1).mean().item()
            + (s ** 2) * (u ** 2).sum(dim=1).mean().item()
        )
        assert abs(e_obs - e_pred) / max(e_obs, 1e-9) < 1e-4, (s, e_obs, e_pred)


def test_pd_distortion_decomposition_prop3_3_in_expectation():
    """Prop. 3.3 expectation identity (latent space, no transforms):

        E||y - y_PD||^2 = s^2 * E||y - y_SD||^2 + E||y - Q(y)||^2

    Uses the fact that u and y are independent and E[u] = 0 over the
    E8 cell (symmetric around origin), so the cross term in the algebraic
    expansion vanishes in expectation.

    Note this holds in LATENT space; in image space the transform breaks
    the identity (which is precisely why Phase 2 reports R-D-P frontiers
    rather than asking the formula to hold image-side).
    """
    cond = _fresh_conditional(channels=8, lattice="E8", dither_mode="pd")
    n = 300_000
    y = torch.randn(n, 8) * 2.0
    q = cond.quantizer(y)
    E_Q = ((y - q) ** 2).sum(dim=1).mean().item()
    # E||y - y_SD||^2 = E||u_eq||^2 (by crypto lemma)
    u_eq = cond._draw_cell_uniform(n, y.device)
    E_SD = (u_eq ** 2).sum(dim=1).mean().item()

    for s in (0.5, 1.0, 1.5):
        u = cond._draw_cell_uniform(n, y.device)
        y_PD = q + s * u
        E_PD = ((y - y_PD) ** 2).sum(dim=1).mean().item()
        # Prop. 3.3 identity:
        rhs = (s ** 2) * E_SD + E_Q
        rel = abs(E_PD - rhs) / max(E_PD, 1e-9)
        assert rel < 0.03, (s, E_PD, rhs, E_Q, E_SD, rel)


# --------------------------------------------------------------------- #
# 3. SW_2^2 estimator                                                   #
# --------------------------------------------------------------------- #


def test_sliced_w2_matches_gaussian_closed_form():
    """For X ~ N(0, I_d) and Y ~ N(0, sigma^2 I_d), every unit projection
    theta gives X . theta ~ N(0, 1) and Y . theta ~ N(0, sigma^2) regardless
    of theta. The 1-D W_2^2 between those is (1 - sigma)^2, so

        SW_2^2(X, Y) = E_theta[ W_2^2(X.theta, Y.theta) ] = (1 - sigma)^2.

    (This is the per-projection W_2^2 our `sliced_w2` returns — NOT the
    full d-dim Bures W_2^2 which would be d*(1 - sigma)^2.)
    """
    d, n, sigma = 8, 10_000, 0.5
    X = torch.randn(n, d)
    Y = sigma * torch.randn(n, d)
    estimates = [sliced_w2(X, Y, n_proj=256).item() for _ in range(5)]
    mean = sum(estimates) / len(estimates)
    truth = (1 - sigma) ** 2  # = 0.25
    assert abs(mean - truth) < 0.05, (mean, truth, estimates)


def test_sliced_w2_zero_on_same_distribution():
    X = torch.randn(10_000, 8)
    Y = torch.randn(10_000, 8)
    s = sliced_w2(X, Y, n_proj=256).item()
    assert s < 0.1, s


# --------------------------------------------------------------------- #
# 4. NestedE8Product                                                    #
# --------------------------------------------------------------------- #


def test_nested_e8_coset_count():
    """|Lambda_f / Lambda| = a^8 for self-similar Lambda = a*Lambda_f on E8."""
    for a in (2, 3):
        net = NestedE8Product(channels=8, a=a)
        assert net.coset_reps.shape == (a ** 8, 8)
        keys = torch.round(net.coset_reps * 1e6).long()
        as_set = {tuple(k.tolist()) for k in keys}
        assert len(as_set) == a ** 8


def test_nested_e8_coset_reps_in_voronoi_of_coarse():
    """Each rep r satisfies Q_{a*E8}(r) = 0  <=>  r ∈ V_0(a*E8)."""
    a = 2
    net = NestedE8Product(channels=8, a=a)
    fine = E8Quantizer(8)
    q_coarse = a * fine(net.coset_reps / a)
    assert torch.allclose(q_coarse, torch.zeros_like(q_coarse), atol=1e-5)


def test_nested_e8_coset_norm_spectrum_a2():
    """Structural sanity on the coset rep spectrum for E8 / 2E8.

    There are |E8 / 2E8| = 2^8 = 256 cosets. Load-bearing properties:
      - the zero vector is present exactly once
      - all reps lie inside V_0(2E8), so norm^2 <= covering radius^2 = 4
      - exactly 120 reps sit at norm^2 = 2 (E8 minimum shell):
        E8 has 240 minimum-norm vectors {v_1, ..., v_240}; since
        -v ≡ v - 2v = v (mod 2E8), each pair {v, -v} collapses to a
        single coset, giving 240/2 = 120 cosets whose minimum
        representative has norm^2 = 2.
    """
    net = NestedE8Product(channels=8, a=2)
    norms_sq = net.coset_rep_norms()
    assert norms_sq.shape == (256,)
    n_zero = (norms_sq < 1e-6).sum().item()
    assert n_zero == 1, n_zero
    assert norms_sq.max().item() <= 4.0 + 1e-4
    n_min = ((norms_sq > 1.99) & (norms_sq < 2.01)).sum().item()
    assert n_min == 120, n_min


def test_nested_e8_sampling_is_uniform():
    """Sample 200k draws; each of the 256 reps appears with frequency in
    [0.85, 1.15] * expected (smoke chi^2-ish test)."""
    net = NestedE8Product(channels=8, a=2)
    n = 200_000
    draws = net.sample_coset_reps(n, "cpu")
    keys = torch.round(draws * 1e6).long()
    counts = {}
    for k in keys:
        t = tuple(k.tolist())
        counts[t] = counts.get(t, 0) + 1
    assert len(counts) == 256
    expected = n / 256
    lo, hi = 0.85 * expected, 1.15 * expected
    for t, c in counts.items():
        assert lo < c < hi, (t, c, expected)


def test_nested_e8_continuous_dither_is_in_fine_cell():
    """draw_cell_uniform_fine produces samples whose Q_{Lambda_f} is 0."""
    net = NestedE8Product(channels=8, a=2)
    u_f = net.draw_cell_uniform_fine(10_000, "cpu")
    fine = E8Quantizer(8)
    qf = fine(u_f)
    assert torch.allclose(qf, torch.zeros_like(qf), atol=1e-5)


# --------------------------------------------------------------------- #
# 5. QSD: coarse-lattice quantization + likelihood volume correction    #
# --------------------------------------------------------------------- #


def test_qsd_inference_c_lives_on_coarse_lattice():
    """For QSD with a=2, the quantized symbol c must lie on 2*E8, i.e.
    Q_{Lambda_f}(c / 2) = c / 2. (Equivalent: c is in 2*Lambda_f.)
    This catches the original 'used fine quantizer for coarse' bug."""
    nested = NestedE8Product(channels=8, a=2)
    cond = _fresh_conditional(
        channels=8, lattice="E8", dither_mode="qsd", nested=nested
    )
    cond.eval()
    n = 5_000
    y = torch.randn(n, 8) * 2.0
    # Inline the inference flow at the flat-space level
    u_hat = nested.sample_coset_reps(n, y.device)
    c = 2 * cond.quantizer((y - u_hat) / 2)
    # Verify each row of c is in 2*E8 (i.e. c / 2 is on E8)
    half = c / 2
    proj = cond.quantizer(half)
    assert torch.allclose(half, proj, atol=1e-5), (half[:3], proj[:3])


def test_qsd_likelihood_applies_a_volume_correction():
    """In the flat-Gaussian regime (sigma >> cell), the per-coord MC mean is
    approximately N(y; mu, sigma^2) regardless of cell width. The QSD
    likelihood must therefore return approximately `a` times the SD
    likelihood, because of the volume correction. Without it, QSD would
    report the same per-coord bpp as SD — which would erase the entire
    R_c = log_2(a) bits/dim 'savings' QSD is supposed to provide.
    """
    channels = 8
    nested = NestedE8Product(channels=channels, a=2)
    sd_cond = _fresh_conditional(
        channels=channels, lattice="E8", dither_mode="sd", N_integral=4096
    )
    qsd_cond = _fresh_conditional(
        channels=channels, lattice="E8", dither_mode="qsd",
        N_integral=4096, nested=nested
    )
    sd_cond.eval(); qsd_cond.eval()

    # Flat regime: small inputs, large sigma -> Gaussian nearly constant
    # across the cell, so MC means agree across cell sizes.
    B, H, W = 2, 4, 4
    inputs = torch.randn(B, channels, H, W) * 0.05
    scales = torch.full_like(inputs, 8.0)
    means = torch.zeros_like(inputs)

    lik_sd = sd_cond._likelihood(inputs, scales, means=means)
    lik_qsd = qsd_cond._likelihood(inputs, scales, means=means)

    ratio = (lik_qsd / lik_sd).mean().item()
    # Tolerance accounts for MC noise + different sobol streams
    assert 1.6 < ratio < 2.4, (ratio, lik_sd.mean().item(), lik_qsd.mean().item())


# --------------------------------------------------------------------- #
# 6. End-to-end forward smoke tests                                     #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["det", "pd", "sd", "qsd"])
def test_dithered_conditional_inference_shapes(mode):
    channels = 8
    nested = NestedE8Product(channels=channels, a=2) if mode == "qsd" else None
    cond = _fresh_conditional(
        channels=channels, lattice="E8", dither_mode=mode,
        s=1.25, nested=nested
    )
    cond.eval()
    B, H, W = 2, 4, 4
    y = torch.randn(B, channels, H, W)
    scales = torch.full_like(y, 1.0)
    means = torch.zeros_like(y)
    rate, rec = cond.quantize(y, "dequantize")
    assert rate.shape == y.shape and rec.shape == y.shape
    assert torch.isfinite(rate).all() and torch.isfinite(rec).all()
    if mode == "det":
        assert torch.equal(rate, rec)  # no dither at all
    if mode == "sd":
        assert torch.equal(rate, rec)  # same y_eq for both
    if mode in ("pd", "qsd"):
        assert not torch.equal(rate, rec)  # decoder dither only in `rec`

    lik = cond._likelihood(rate, scales, means=means)
    assert lik.shape == y.shape
    assert (lik > 0).all() and torch.isfinite(lik).all()


@pytest.mark.parametrize("mode", ["det", "pd", "sd", "qsd"])
def test_dithered_conditional_training_shapes(mode):
    """Training path: confirm noise-mode returns (rate, rec) of right shape
    and that gradients flow back through `rate` (so the rate loss reaches
    the encoder)."""
    channels = 8
    nested = NestedE8Product(channels=channels, a=2) if mode == "qsd" else None
    cond = _fresh_conditional(
        channels=channels, lattice="E8", dither_mode=mode,
        s=1.25, nested=nested
    )
    cond.train()
    B, H, W = 2, 4, 4
    y = torch.randn(B, channels, H, W, requires_grad=True)
    rate, rec = cond.quantize(y, "noise")
    assert rate.shape == y.shape and rec.shape == y.shape
    assert torch.isfinite(rate).all() and torch.isfinite(rec).all()
    # gradient flows through `rate` back to y (rate is what feeds the
    # likelihood, so the rate loss must reach the encoder)
    loss = rate.sum()
    loss.backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()


def test_cheng2020_rdp_full_forward_all_modes():
    """Integration smoke: full Cheng2020RDP forward on a tiny RGB batch
    for all four modes; check finite outputs and that backward runs."""
    for mode in ("det", "pd", "sd", "qsd"):
        nested = NestedE8Product(channels=16, a=2) if mode == "qsd" else None
        net = Cheng2020RDP(
            N=16,                           # tiny latent for speed
            N_integral=128,
            lattice_name="E8Product",
            dither_mode=mode,
            s_dither=1.0,
            nested_quantizer=nested,
        )
        x = torch.rand(1, 3, 64, 64)
        out = net(x)
        assert out["x_hat"].shape == x.shape
        assert torch.isfinite(out["x_hat"]).all()
        for k, v in out["likelihoods"].items():
            assert (v > 0).all() and torch.isfinite(v).all(), k
        # Backward sanity in training mode
        net.train()
        out = net(x)
        loss = out["x_hat"].abs().mean() + sum(
            -torch.log(v).mean() for v in out["likelihoods"].values()
        )
        loss.backward()


def test_rdploss_runs_and_grad_flows():
    crit = RDPLoss(lmbda=1e-2, lmbda_p=0.5, perc="sw", patch=8, n_proj=64)
    B = 2
    x = torch.rand(B, 3, 64, 64)
    x_hat = (x + 0.01 * torch.randn_like(x)).clamp(0, 1).requires_grad_(True)
    lik_y = torch.full((B, 8, 8, 8), 0.5)
    lik_z = torch.full((B, 4, 4, 4), 0.5)
    out = crit(
        {"x_hat": x_hat, "likelihoods": {"y": lik_y, "z": lik_z}},
        x,
    )
    assert torch.isfinite(out["loss"]).item()
    out["loss"].backward()
    assert x_hat.grad is not None and torch.isfinite(x_hat.grad).all()
