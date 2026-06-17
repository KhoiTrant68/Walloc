"""
B4 baseline: vanilla cheng2020-attn on Kodak.

Runs CompressAI's pretrained cheng2020-attn at user-selected quality
levels (default q=1..6) on the test set. Reports per-image bpp / PSNR /
SW2 (same metrics as scripts/eval.py) so the WALLOC R-D-P curve can be
overlaid against a published reference.

This is an EVAL-ONLY script. No training. The model is downloaded from
CompressAI's S3 bucket on first use and cached in ~/.cache/torch.

Stand-alone example:

    python scripts/baseline_cheng.py \
        --qualities 1,2,3,4,5,6 \
        --test_root /kaggle/input/kodak-test \
        --out_json /kaggle/working/walloc/baselines/cheng2020attn.json

Output JSON shape mirrors scripts/eval.py but with one entry per quality:

    {
      "model": "cheng2020-attn",
      "metric": "mse",
      "results": [
        {"quality": 1, "summary": {...}, "per_image": [...]},
        ...
      ]
    }

TODO (B4 follow-up): add a HiFiC baseline. HiFiC is a perception-aware
codec and is the right external comparison for WALLOC's RDP curve. It is
not in CompressAI; the reference implementation is at
https://github.com/tensorflow/compression/tree/master/models/hific
and would need to be wrapped in a separate runner (different framework,
different checkpoint format). Out of scope for this script.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from LTC.perception import extract_patches, sliced_w2  # noqa: E402
from scripts.data import IMAGE_EXTS, autodetect_dataset  # noqa: E402


def list_images(root: str):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(Path(root).rglob(f"*{ext}"))
    paths = sorted(set(str(p) for p in paths))
    if not paths:
        raise FileNotFoundError(f"no images under {root}")
    return paths


def pad_to_multiple(x: torch.Tensor, m: int = 64) -> torch.Tensor:
    _, h, w = x.shape
    H = ((h + m - 1) // m) * m
    W = ((w + m - 1) // m) * m
    return F.pad(x, (0, W - w, 0, H - h), mode="replicate")


def bpp_from_likelihoods(out_net, num_pixels: int) -> float:
    s = 0.0
    for lik in out_net["likelihoods"].values():
        s += float(torch.log(lik).sum() / (-math.log(2) * num_pixels))
    return s


@torch.no_grad()
def evaluate_quality(quality: int, paths, device):
    from compressai.zoo import image_models as pretrained_models

    print(f"\n[baseline] cheng2020-attn q={quality}")
    net = pretrained_models["cheng2020-attn"](
        quality=quality, metric="mse", pretrained=True, progress=False,
    ).to(device).eval()

    tf = transforms.ToTensor()
    rows = []
    for p in paths:
        img = tf(Image.open(p).convert("RGB"))
        _, h, w = img.shape
        x = pad_to_multiple(img).unsqueeze(0).to(device)
        out = net(x)
        x_hat = out["x_hat"].clamp(0, 1)
        x_orig = img.unsqueeze(0).to(device)
        x_hat_crop = x_hat[..., :h, :w]
        npix = x_orig.shape[0] * h * w
        mse = F.mse_loss(x_hat_crop, x_orig).item()
        psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item()
        bpp = bpp_from_likelihoods(out, npix)
        sw = sliced_w2(
            extract_patches(x_orig, patch=8, max_patches=4096),
            extract_patches(x_hat_crop, patch=8, max_patches=4096),
            n_proj=128,
        ).item()
        rows.append({
            "name": os.path.basename(p),
            "h": h, "w": w,
            "bpp": bpp, "psnr": psnr, "mse": mse, "sw2_patch": sw,
        })
        print(f"  {rows[-1]['name']:24s} bpp={bpp:.4f} psnr={psnr:.3f} "
              f"sw2={sw:.3e}")

    keys = ("bpp", "psnr", "mse", "sw2_patch")
    summary = {f"{k}_mean": sum(r[k] for r in rows) / len(rows) for k in keys}
    summary["n_images"] = len(rows)
    return {"quality": quality, "summary": summary, "per_image": rows}


def main(argv=None):
    p = argparse.ArgumentParser("WALLOC baseline: cheng2020-attn")
    p.add_argument("--qualities", default="1,2,3,4,5,6",
                   help="comma-separated quality levels (1..6 for cheng2020-attn)")
    p.add_argument("--test_root", default=os.environ.get("WALLOC_TEST_INPUT"))
    p.add_argument("--out_json", required=True)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.test_root = autodetect_dataset(args.test_root, purpose="test")
    paths = list_images(args.test_root)
    print(f"[baseline] {len(paths)} images from {args.test_root}")

    qualities = [int(q) for q in args.qualities.split(",") if q.strip()]
    results = []
    t0 = time.time()
    for q in qualities:
        results.append(evaluate_quality(q, paths, device))
    elapsed = time.time() - t0

    out = {
        "model": "cheng2020-attn",
        "metric": "mse",
        "test_root": args.test_root,
        "time_s": elapsed,
        "results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE R-D CURVE: cheng2020-attn on test set")
    print("=" * 60)
    print(f"{'quality':>8}  {'bpp':>8}{'psnr':>8}{'sw2':>11}")
    for r in results:
        s = r["summary"]
        print(f"{r['quality']:>8}  {s['bpp_mean']:>8.4f}{s['psnr_mean']:>8.3f}"
              f"{s['sw2_patch_mean']:>11.3e}")
    print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
