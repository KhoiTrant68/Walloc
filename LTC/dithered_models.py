"""
WALLOC — Wasserstein-Anchored Lattice Codec.

Dithered extension of GaussianConditionalLattice / Cheng2020AttentionLattice
implementing the three operational modes of Lei et al. arXiv 2503.17558v2:

    - 'det' : deterministic LTC (baseline, R_c = 0, no randomness)
    - 'pd'  : Private Dither LTC (R_c = 0, dither only at decoder; the rate
              path uses noisy-proxy for tractability — Fig. 12 in the paper
              shows STE vs. noisy-proxy is nearly indistinguishable for PD)
    - 'sd'  : Shared Dither LTC (R_c = inf, synchronized dither at both sides)
    - 'qsd' : Quantized Shared Dither (R_c = log_2(a) bits/dim, discrete
              dither u_hat ~ Unif(Lambda_f / Lambda) with Lambda = a * Lambda_f)

Design contract — `quantize(...)` always returns the pair

        (rate_input, rec_input)

for both training (mode='noise') and inference (mode='dequantize').
`rate_input` is what the context model and the entropy likelihood see;
`rec_input` is what g_s sees. This avoids the "is this c or c+u?" ambiguity
that plagued the first cut of the code and matches the paper's eq. (4)/(6)
exactly:

                    rate_input          rec_input               operational rate
        det (train) y + u (noisy proxy) STE(Q(y))               H(Q(y)) approx by noisy proxy
        det (eval)  Q(y)                Q(y)                    H(Q(y))
        pd  (train) y + u (noisy proxy) STE(Q(y)) + s * u       H(Q(y)) approx by noisy proxy (Fig. 12)
        pd  (eval)  Q(y)                Q(y) + s * u            H(Q(y))
        sd  (train) y + u               y + u                   H(c|u) via eq. (6) (4)==(6) for SD
        sd  (eval)  Q(y-u) + u          Q(y-u) + u              H(c|u) via eq. (6); =d y + u_eq (crypto lemma)
        qsd (train) c + u_hat (STE)     c + u_hat + s * u_f     H(c|u_hat) via eq. (4) on coarse cell
        qsd (eval)  c + u_hat           c + u_hat + s * u_f     H(c|u_hat)

For SD, training rate == reconstruction (both = y + u). That is intentional:
the crypto lemma Q_Lambda(y - u) + u =d y + u_eq makes this both the eq. (6)
operational-rate target AND distribution-matched to the inference-time
reconstruction. No train/test mismatch — this is what made SD-LTC fit so
cleanly into a noise-substitution training pipeline in the first place.

For QSD the likelihood must integrate over V_0(a*Lambda_f), not V_0(Lambda_f),
and a per-coord volume correction of factor `a` is applied (under the
upstream code's per-coord product-likelihood approximation). See
`_likelihood`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from LTC.models_compressai import (
    Cheng2020AttentionLattice,
    GaussianConditionalLattice,
)


class DitheredGaussianConditionalLattice(GaussianConditionalLattice):
    """Adds dither_mode in {'det','pd','sd','qsd'} on top of the parent.

    See module docstring for the exact training / inference contract.
    """

    def __init__(
        self,
        *args,
        dither_mode: str = "sd",
        s_dither: float = 1.0,
        nested_quantizer=None,  # only used for 'qsd' — instance of NestedE8Product
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if dither_mode not in ("det", "pd", "sd", "qsd"):
            raise ValueError(f"unknown dither_mode {dither_mode!r}")
        if dither_mode == "qsd" and nested_quantizer is None:
            raise ValueError("dither_mode='qsd' requires a nested_quantizer")
        if nested_quantizer is not None and nested_quantizer.channels != self.channels:
            raise ValueError(
                f"nested_quantizer.channels ({nested_quantizer.channels}) "
                f"must equal channels ({self.channels})"
            )
        self.dither_mode = dither_mode
        # s > 1 trades distortion for perception per Prop. 3.3 / Remark 4.7.
        # Gate sweeps s in {1.0, 1.25, 1.5}; Phase-2 schedules s(lambda).
        self.s_dither = float(s_dither)
        self.nested_quantizer = nested_quantizer

    # ------------------------------------------------------------------ #
    # internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _draw_cell_uniform(self, n_rows: int, device) -> Tensor:
        """Draw n_rows samples ~ Unif(V_0(Lambda_f)) using the parent's
        Sobol engine. Mapping (sobol -> G -> mod V_0) matches the convention
        used by `_likelihood`, so MC integration and dither share one
        quasi-random stream — important to keep them consistent."""
        u = self.sobol_eng.draw(n_rows).to(device)
        u = u @ self.quantizer.G.to(device)
        return u - self.quantizer(u)

    # ------------------------------------------------------------------ #
    # quantize: returns (rate_input, rec_input) for ALL modes            #
    # ------------------------------------------------------------------ #

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if mode == "noise":
            return self._noise_quantize(inputs, means)
        if mode == "dequantize":
            return self._inference_quantize(inputs, means)
        if mode == "symbols":
            ri, _ = self._inference_quantize(inputs, means)
            return ri
        raise ValueError(f"unknown quantize mode {mode!r}")

    # ----- training paths --------------------------------------------- #

    def _noise_quantize(self, inputs: Tensor, means: Optional[Tensor]):
        if self.dither_mode == "qsd":
            return self._qsd_noise(inputs)
        # det / sd / pd: re-use parent's noise path which produces
        #   y_tilde = y + u_fine   (u_fine ~ Unif(V_0(Lambda_f)))
        #   y_hat   = STE(Q_{Lambda_f}(y))
        y_tilde, y_hat = super().quantize(inputs, "noise", means)
        if self.dither_mode == "det":
            return y_tilde, y_hat  # noisy-proxy rate, STE reconstruction
        if self.dither_mode == "sd":
            # SD: rate = E_u[-log E_u'[N(y+u+u')]] on y_tilde = y + u, AND
            # reconstruction is distribution-matched to inference (=d y + u_eq).
            # Returning y_tilde for both ties training to inference exactly,
            # not just in distribution.
            return y_tilde, y_tilde
        # PD: noisy-proxy rate on y_tilde (Fig. 12 of the paper),
        # decoder-side dither s*u baked into reconstruction so g_s learns to
        # denoise it. Using detach() so the dither is treated as data noise.
        u = (y_tilde - inputs).detach()
        return y_tilde, y_hat + self.s_dither * u

    def _qsd_noise(self, inputs: Tensor):
        """QSD training: c = Q_{a*Lambda_f}(y - u_hat) with STE; rate input
        is c + u_hat; reconstruction adds the continuous fine dither s*u_f."""
        nq = self.nested_quantizer
        a = nq.a
        y, y_shape = self._flatten(inputs)
        u_hat = nq.sample_coset_reps(y.shape[0], y.device)
        u_f = nq.draw_cell_uniform_fine(y.shape[0], y.device)
        diff = y - u_hat
        # Q_{a*Lambda_f}(x) = a * Q_{Lambda_f}(x / a)  (self-similar nesting)
        c_hard = a * self.quantizer(diff / a)
        # STE: forward value = c_hard, backward = identity on `diff` (i.e. on y).
        c = diff + (c_hard - diff).detach()
        rate = c + u_hat  # = y + (c_hard - diff).detach()  -> grad flows to y
        rec = c + u_hat + self.s_dither * u_f
        return self._unflatten(rate, y_shape), self._unflatten(rec, y_shape)

    # ----- inference paths -------------------------------------------- #

    def _inference_quantize(self, inputs: Tensor, means: Optional[Tensor]):
        outputs = inputs.clone()
        if means is not None:
            outputs = outputs - means
        y, y_shape = self._flatten(outputs)

        if self.dither_mode == "det":
            c = self.quantizer(y)
            rate, rec = c, c
        elif self.dither_mode == "pd":
            c = self.quantizer(y)
            u = self._draw_cell_uniform(y.shape[0], y.device)
            rate, rec = c, c + self.s_dither * u
        elif self.dither_mode == "sd":
            u = self._draw_cell_uniform(y.shape[0], y.device)
            c = self.quantizer(y - u)
            # crypto lemma: c + u =d y + u_eq with u_eq ~ Unif(V_0(Lambda_f))
            rate = rec = c + u
        elif self.dither_mode == "qsd":
            nq = self.nested_quantizer
            a = nq.a
            u_hat = nq.sample_coset_reps(y.shape[0], y.device)
            u_f = nq.draw_cell_uniform_fine(y.shape[0], y.device)
            # c on the COARSE lattice a*Lambda_f
            c = a * self.quantizer((y - u_hat) / a)
            rate = c + u_hat
            rec = c + u_hat + self.s_dither * u_f
        else:  # pragma: no cover — guarded in __init__
            raise RuntimeError(self.dither_mode)

        rate = self._unflatten(rate, y_shape)
        rec = self._unflatten(rec, y_shape)
        if means is not None:
            rate = rate + means
            rec = rec + means
        return rate, rec

    # ------------------------------------------------------------------ #
    # likelihood — for QSD we integrate over V_0(a*Lambda_f) instead of  #
    # V_0(Lambda_f) AND multiply by the per-coord volume factor `a` so   #
    # the returned lik approximates prob_i = a * MC_mean_i under the     #
    # per-coord product approximation that the upstream pipeline uses.   #
    # ------------------------------------------------------------------ #

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        if self.dither_mode != "qsd" or self.nested_quantizer is None:
            return super()._likelihood(inputs, scales, means)
        a = self.nested_quantizer.a
        scales = self.lower_bound_scale(scales)
        # MC noise drawn from V_0(Lambda_f), scaled by `a` -> V_0(a*Lambda_f).
        # Self-similar: V_0(a*L) = a * V_0(L).
        u = self.sobol_eng.draw(self.N_integral).to(inputs.device)
        u2 = u @ self.quantizer.G
        u2 = u2 - self.quantizer(u2)
        u2 = u2 * a
        y, y_shape = self._flatten(inputs)
        scales_flatten, _ = self._flatten(scales)
        if means is None:
            means_flatten = torch.zeros_like(scales_flatten)
        else:
            means_flatten, _ = self._flatten(means)
        lik = self._lik_MC_est(y, u2, scales_flatten, means_flatten)
        # Per-coord volume correction under the product approximation:
        # prob_i(coset) ≈ a · E_{u_i ~ Unif(scaled cell)}[ N(y_i + u_i) ].
        # Without this factor we would over-report the rate by log_2(a)
        # bits per latent coordinate — the entire R_c "savings" of QSD.
        lik = a * lik
        return self._unflatten(lik, y_shape)


# ---------------------------------------------------------------------- #
# Backbone wrapper                                                       #
# ---------------------------------------------------------------------- #


class Cheng2020RDP(Cheng2020AttentionLattice):
    """Cheng2020Attention with a DitheredGaussianConditionalLattice.

    The forward picks `rate_input` and `rec_input` from the dithered
    conditional and routes them through the existing hyperprior + context
    + Gaussian-conditional likelihood pipeline.
    """

    def __init__(
        self,
        N: int = 128,
        N_integral: int = 2048,
        lattice_name: str = "E8Product",
        dither_mode: str = "sd",
        s_dither: float = 1.0,
        nested_quantizer=None,
        **kwargs,
    ):
        super().__init__(
            N=N, N_integral=N_integral, lattice_name=lattice_name, **kwargs
        )
        # Replace the parent's conditional with the dithered version.
        self.gaussian_conditional = DitheredGaussianConditionalLattice(
            None,
            channels=N,
            N_integral=N_integral,
            lattice_name=lattice_name,
            dither_mode=dither_mode,
            s_dither=s_dither,
            nested_quantizer=nested_quantizer,
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        gc = self.gaussian_conditional
        mode = "noise" if self.training else "dequantize"
        rate_input, rec_input = gc.quantize(y, mode)

        ctx_params = self.context_prediction(rate_input)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_likelihoods = gc._likelihood(rate_input, scales_hat, means=means_hat)
        if gc.use_likelihood_bound:
            y_likelihoods = gc.likelihood_lower_bound(y_likelihoods)
        x_hat = self.g_s(rec_input)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
