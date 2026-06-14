"""
Nested lattice sampler for QSD-LTC (Quantized Shared Dither).

Implements coset-rep sampling for a self-similar nested pair
    Lambda_coarse = a * Lambda_fine
with Lambda_fine = E8 (or E8 product). Used by `dithered_models.py` when
`dither_mode == 'qsd'`.

Number of cosets = a^n per E8 block. The shared discrete dither u_hat is
drawn uniformly from these a^n reps; an additional continuous dither u_f ~
Unif(V_0(Lambda_fine)) is drawn at the decoder (paper eq. for QSD-LTC).

Operational rate of communicating u_hat is R_c = log_2(a) bits per latent
coordinate (the "shared randomness budget"). Setting a=2 gives 1 bit/dim,
a=3 gives log2(3) approx 1.585 bit/dim. The paper observes a=3 already
gets close to SD-LTC for the toy sources.

We DO NOT claim seed-based PRNG = free infinite randomness. The whole point
of QSD is that R_c is finite and quantifiable. See Remark 3.6 in the paper.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from LTC.quantizers import E8Quantizer


class NestedE8Product(nn.Module):
    """Self-similar nested E8 product: Lambda_coarse = a * Lambda_fine,
    where Lambda_fine is the E8 product on `channels` coordinates (channels
    must be a multiple of 8).

    At init we enumerate the a^8 coset reps of E8 / (a*E8) once, store them
    as a buffer of shape [a^8, 8], and reuse for all blocks. Sampling a
    dither for a batch of `n_rows` latent vectors of dim `channels` =
    `n_blocks * 8` costs an int draw + a gather.
    """

    def __init__(self, channels: int, a: int = 2):
        super().__init__()
        if channels % 8 != 0:
            raise ValueError(f"channels {channels} not divisible by 8")
        if a < 2:
            raise ValueError("nesting ratio a must be >= 2")
        self.channels = int(channels)
        self.n_blocks = self.channels // 8
        self.a = int(a)
        self.n_cosets = self.a**8  # |Lambda_f / Lambda| per block

        # fine and coarse E8 (single block); reuse for every block
        self._fine = E8Quantizer(8)
        # coset reps in V_0(a*E8), shape [a^8, 8]
        reps = self._enumerate_coset_reps()
        self.register_buffer("coset_reps", reps)  # float32
        # NSM (continuous) of the fine cell, for sanity-checking u_f draws
        # not used inside the loss; just exposed.
        self.register_buffer(
            "_fine_G", self._fine.G.detach().clone()
        )

    # ------------------------------------------------------------------ #
    # coset-rep enumeration (one-time precompute)                        #
    # ------------------------------------------------------------------ #

    def _enumerate_coset_reps(self) -> Tensor:
        """Build the a^8 coset reps of E8 / (a*E8) by enumerating all
        digit-combinations of the E8 basis (mod a) and reducing each into
        V_0(a*E8). With a=2 this gives exactly 2^8 = 256 unique reps; we
        assert the count to catch implementation drift.
        """
        a = self.a
        G = self._fine.G.detach().cpu()  # [8, 8]
        # digit vectors in {0, 1, ..., a-1}^8
        ranges = [torch.arange(a, dtype=torch.float)] * 8
        digits = torch.cartesian_prod(*ranges)  # [a^8, 8]
        # candidate fine-lattice points in V_0(a*E8): sum d_i * b_i then mod (a*Lambda_f)
        cands = digits @ G  # [a^8, 8]
        # Q_{a*Lambda_f}(x) = a * Q_{Lambda_f}(x/a)
        q_coarse = a * self._fine(cands / a)
        reps = cands - q_coarse  # canonical "closest to origin" reps

        # dedupe with rounding tolerance (lattice coords are exact up to fp noise)
        keys = torch.round(reps * 1e6).long()
        seen = {}
        out = []
        for i in range(reps.shape[0]):
            k = tuple(keys[i].tolist())
            if k not in seen:
                seen[k] = True
                out.append(reps[i])
        out = torch.stack(out, dim=0)
        if out.shape[0] != self.n_cosets:
            raise RuntimeError(
                f"coset enumeration failed: got {out.shape[0]} unique reps, "
                f"expected {self.n_cosets} (a={a}). "
                "Try increasing the enumeration support or check the E8 generator."
            )
        return out

    # ------------------------------------------------------------------ #
    # sampling                                                           #
    # ------------------------------------------------------------------ #

    def sample_coset_reps(self, n_rows: int, device) -> Tensor:
        """Return a tensor [n_rows, channels] whose every 8-block is an
        independent uniform draw from the a^8 coset reps. This is the
        shared discrete dither u_hat ~ Unif(Lambda_f / Lambda).
        """
        idx = torch.randint(
            0, self.n_cosets, (n_rows, self.n_blocks), device=device
        )  # [n_rows, n_blocks]
        reps = self.coset_reps.to(device)  # [n_cosets, 8]
        out = reps[idx]  # [n_rows, n_blocks, 8]
        return out.reshape(n_rows, self.channels)

    def draw_cell_uniform_fine(self, n_rows: int, device) -> Tensor:
        """Continuous fine-lattice dither u_f ~ Unif(V_0(Lambda_fine)). Used
        only at the decoder. We draw via the standard sobol -> G_fine -> mod
        V_0 trick, but with vanilla torch.rand here to avoid coupling to the
        likelihood's MC Sobol stream.
        """
        # Per-block uniform in the fine fundamental parallelepiped, then mod cell
        z = torch.rand(n_rows * self.n_blocks, 8, device=device)
        Gf = self._fine_G.to(device)
        w = z @ Gf
        u = w - self._fine(w)
        return u.reshape(n_rows, self.channels)

    # ------------------------------------------------------------------ #
    # diagnostics                                                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def coset_rep_norms(self) -> Tensor:
        """For sanity-checking: squared norms of the enumerated coset reps.
        For E8 / 2E8 the reps include 0 (norm 0), the 240 minimal vectors
        of E8 (norm^2 = 2), and 15 others of norm^2 = 4 (total 256).
        """
        return (self.coset_reps**2).sum(dim=1)

    @property
    def R_c_bits_per_dim(self) -> float:
        """log_2(a) bits per latent coordinate — the operational
        shared-randomness budget."""
        return float(np.log2(self.a))
