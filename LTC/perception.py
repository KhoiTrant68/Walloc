"""
WALLOC perception losses and the full R-D-P training criterion.

Perception is measured at the distribution level over image patches:
    delta(P_x, P_x_hat) = SW_2^2(patches(x), patches(x_hat))
(or, optionally, a debiased Sinkhorn divergence). This is the strong-sense
W_2-style perception constraint used throughout the RDP literature (Blau &
Michaeli 2018, Lei et al. 2025). We deliberately avoid per-image distortion
masquerading as perception (LPIPS, MS-SSIM): the paper's claim is on
distributional realism, not per-image similarity.

SW_2^2 is the primary estimator (cheap, matches the proxy used in
arXiv 2503.17558v2 App. E). Debiased Sinkhorn via `geomloss` is provided as a
secondary check — useful for the "gap-to-bound" analysis in the writeup.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------- #
# patch extraction                                                       #
# ---------------------------------------------------------------------- #


def extract_patches(
    img: Tensor, patch: int = 8, max_patches: Optional[int] = 4096
) -> Tensor:
    """Pull non-overlapping `patch x patch` RGB patches from a [B,3,H,W]
    image batch and flatten to [N, 3*patch*patch]. Optionally subsample to
    `max_patches` total patches (random across the batch) to keep the cost
    of SW_2 / Sinkhorn bounded.
    """
    assert img.dim() == 4, img.shape
    p = patch
    x = F.unfold(img, kernel_size=p, stride=p)  # [B, 3*p*p, L]
    x = x.permute(0, 2, 1).reshape(-1, x.shape[1])  # [B*L, 3*p*p]
    if max_patches is not None and x.shape[0] > max_patches:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_patches]
        x = x[idx]
    return x


# ---------------------------------------------------------------------- #
# sliced 2-Wasserstein squared                                           #
# ---------------------------------------------------------------------- #


def sliced_w2(X: Tensor, Y: Tensor, n_proj: int = 128) -> Tensor:
    """SW_2^2(P_X, P_Y) Monte-Carlo estimate.

    X: [N, d], Y: [M, d]. Returns a scalar tensor.

    Implementation:
      - draw d x n_proj random unit directions
      - project X, Y onto each direction
      - the optimal 1-D transport for empirical measures with equal mass is
        sort-then-difference; if N != M we equalise via empirical quantiles
        on a common grid of size min(N, M)
    """
    assert X.dim() == 2 and Y.dim() == 2 and X.shape[1] == Y.shape[1]
    d, device = X.shape[1], X.device
    theta = torch.randn(d, n_proj, device=device, dtype=X.dtype)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
    Xp, Yp = X @ theta, Y @ theta  # [N, P], [M, P]

    if Xp.shape[0] == Yp.shape[0]:
        Xs, _ = torch.sort(Xp, dim=0)
        Ys, _ = torch.sort(Yp, dim=0)
    else:
        K = min(Xp.shape[0], Yp.shape[0])
        q = torch.linspace(0.0, 1.0, K, device=device, dtype=X.dtype)
        Xs = torch.quantile(Xp, q, dim=0)
        Ys = torch.quantile(Yp, q, dim=0)

    return ((Xs - Ys) ** 2).mean()


# ---------------------------------------------------------------------- #
# Sinkhorn divergence (optional)                                         #
# ---------------------------------------------------------------------- #


def sinkhorn_div(X: Tensor, Y: Tensor, blur: float = 0.05) -> Tensor:
    """Debiased Sinkhorn divergence S_eps(X, Y) via geomloss. Returns a
    differentiable scalar in the same units as squared Euclidean distance.
    Requires `pip install geomloss pykeops`.
    """
    from geomloss import SamplesLoss  # lazy import — heavy

    loss = SamplesLoss("sinkhorn", p=2, blur=blur, debias=True)
    return loss(X, Y)


# ---------------------------------------------------------------------- #
# R-D-P criterion                                                        #
# ---------------------------------------------------------------------- #


class RDPLoss(nn.Module):
    """Rate-Distortion-Perception loss:

        L = bpp + lambda * 255^2 * MSE  +  lambda_p * delta(P_x, P_x_hat)

    Rate is computed from the model's reported likelihoods (operational
    cross-entropy / num_pixels). Distortion is MSE on [0,1] images, scaled
    by 255^2 to match CompressAI's convention so `lambda` values are
    comparable to standard image-codec literature. Perception is SW_2^2 on
    patches by default; switch to Sinkhorn via `perc='sinkhorn'`.
    """

    METRICS = ("sw", "sinkhorn")

    def __init__(
        self,
        lmbda: float = 1e-2,
        lmbda_p: float = 0.0,
        perc: str = "sw",
        patch: int = 8,
        n_proj: int = 128,
        max_patches: int = 4096,
        sinkhorn_blur: float = 0.05,
    ):
        super().__init__()
        if perc not in self.METRICS:
            raise ValueError(f"perc must be one of {self.METRICS}, got {perc!r}")
        self.lmbda = float(lmbda)
        self.lmbda_p = float(lmbda_p)
        self.perc = perc
        self.patch = int(patch)
        self.n_proj = int(n_proj)
        self.max_patches = int(max_patches)
        self.sinkhorn_blur = float(sinkhorn_blur)
        self.mse = nn.MSELoss()

    def perception(self, x: Tensor, x_hat: Tensor) -> Tensor:
        Xp = extract_patches(x, self.patch, self.max_patches)
        Yp = extract_patches(x_hat.clamp(0.0, 1.0), self.patch, self.max_patches)
        if self.perc == "sw":
            return sliced_w2(Xp, Yp, self.n_proj)
        return sinkhorn_div(Xp, Yp, blur=self.sinkhorn_blur)

    def forward(self, output, target):
        N, _, H, W = target.size()
        num_pixels = N * H * W
        out = {}
        out["bpp_loss"] = sum(
            (torch.log(lik).sum() / (-math.log(2) * num_pixels))
            for lik in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)

        if self.lmbda_p > 0.0:
            out["perc_loss"] = self.perception(target, output["x_hat"])
        else:
            out["perc_loss"] = torch.zeros((), device=target.device)

        out["loss"] = (
            out["bpp_loss"]
            + self.lmbda * 255.0**2 * out["mse_loss"]
            + self.lmbda_p * out["perc_loss"]
        )
        return out
