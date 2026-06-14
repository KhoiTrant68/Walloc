"""
Evaluate a trained WALLOC checkpoint on an image folder (Kodak / CLIC).

Reports per-image bpp / PSNR / SW_2^2 / LPIPS and aggregates.

Defaults pick up from environment variables written by setup.sh:
    WALLOC_TEST_INPUT    default for --test_root
    WALLOC_OUTPUT_DIR    default base for --out_json (when only filename given)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from LTC.dithered_models import Cheng2020RDP  # noqa: E402
from LTC.nested_lattice import NestedE8Product  # noqa: E402
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
    import math
    s = 0.0
    for lik in out_net["likelihoods"].values():
        s += float(torch.log(lik).sum() / (-math.log(2) * num_pixels))
    return s


def try_lpips(device):
    try:
        import lpips
        net = lpips.LPIPS(net="alex").to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net
    except Exception as e:
        print(f"[lpips] disabled ({e})")
        return None


@torch.no_grad()
def evaluate(net, paths, device, with_lpips=None,
             tb: SummaryWriter | None = None, tb_max_images: int = 8):
    tf = transforms.ToTensor()
    rows = []
    for idx, p in enumerate(paths):
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
        row = {
            "name": os.path.basename(p),
            "h": h, "w": w,
            "bpp": bpp, "psnr": psnr, "mse": mse, "sw2_patch": sw,
        }
        if with_lpips is not None:
            xs = x_orig * 2 - 1
            xhs = x_hat_crop * 2 - 1
            row["lpips"] = with_lpips(xs, xhs).item()
        rows.append(row)
        print(f"  {row['name']:24s} bpp={row['bpp']:.4f} psnr={row['psnr']:.3f} "
              f"sw2={row['sw2_patch']:.3e}"
              + (f" lpips={row['lpips']:.4f}" if 'lpips' in row else ""))

        if tb is not None:
            tb.add_scalar("eval/bpp", row["bpp"], idx)
            tb.add_scalar("eval/psnr", row["psnr"], idx)
            tb.add_scalar("eval/sw2_patch", row["sw2_patch"], idx)
            if "lpips" in row:
                tb.add_scalar("eval/lpips", row["lpips"], idx)
            if idx < tb_max_images:
                xin = img.cpu()
                xrec = x_hat_crop.squeeze(0).cpu()
                tb.add_image(f"eval/{row['name']}/input_vs_recon",
                             make_grid([xin, xrec], nrow=2), 0)
                resid = (xin - xrec).abs()
                resid = resid / resid.amax().clamp_min(1e-6)
                tb.add_image(f"eval/{row['name']}/residual_abs_norm", resid, 0)
    return rows


def aggregate(rows):
    keys = ("bpp", "psnr", "mse", "sw2_patch")
    summary = {f"{k}_mean": sum(r[k] for r in rows) / len(rows) for k in keys}
    if rows and "lpips" in rows[0]:
        summary["lpips_mean"] = sum(r["lpips"] for r in rows) / len(rows)
    summary["n_images"] = len(rows)
    return summary


def build_model(args: dict, device):
    nested = (
        NestedE8Product(channels=args["channels"], a=args["qsd_a"])
        if args["dither_mode"] == "qsd" else None
    )
    net = Cheng2020RDP(
        N=args["channels"], N_integral=args["N_integral"],
        lattice_name=args["lattice_name"],
        dither_mode=args["dither_mode"], s_dither=args["s_dither"],
        nested_quantizer=nested,
    )
    return net.to(device)


def main(argv=None):
    p = argparse.ArgumentParser("WALLOC eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_root", default=os.environ.get("WALLOC_TEST_INPUT"))
    p.add_argument("--out_json", required=True)
    p.add_argument("--lpips", action="store_true", default=True,
                   help="compute LPIPS if package is installed")
    p.add_argument("--tb_dir", default=None,
                   help="optional TensorBoard log dir for per-image metrics + previews")
    p.add_argument("--tb_max_images", type=int, default=8)
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.test_root = autodetect_dataset(args.test_root, purpose="test")

    ck = torch.load(args.checkpoint, map_location=device)
    train_args = ck["args"]
    net = build_model(train_args, device)
    net.load_state_dict(ck["state_dict"])
    net.eval()

    paths = list_images(args.test_root)
    print(f"[eval] {args.checkpoint} on {len(paths)} images from {args.test_root}")

    lpips_net = try_lpips(device) if args.lpips else None
    tb = None
    if args.tb_dir:
        Path(args.tb_dir).mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(log_dir=args.tb_dir)
        print(f"[tb] writing to {args.tb_dir}")
    t0 = time.time()
    rows = evaluate(net, paths, device, with_lpips=lpips_net,
                    tb=tb, tb_max_images=args.tb_max_images)
    summary = aggregate(rows)
    summary["time_s"] = time.time() - t0
    if tb is not None:
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                tb.add_scalar(f"eval_summary/{k}", v, 0)
        tb.flush()
        tb.close()

    out = {
        "tag": train_args.get("tag") or os.path.basename(os.path.dirname(args.checkpoint)),
        "train_args": train_args,
        "per_image": rows,
        "summary": summary,
        "checkpoint": args.checkpoint,
        "test_root": args.test_root,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[summary] {out['tag']}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"-> {args.out_json}")
    return out


if __name__ == "__main__":
    main()
