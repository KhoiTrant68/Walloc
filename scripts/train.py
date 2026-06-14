"""
Train a single WALLOC configuration.

Defaults pick up from environment variables written by setup.sh:
    WALLOC_TRAIN_INPUT   default for --data_root
    WALLOC_TEST_INPUT    default for --test_root
    WALLOC_OUTPUT_DIR    default for --out_dir
    WALLOC_NUM_GPUS      default for --num_gpus (auto if unset)

CLI override always wins. The script multi-GPU-wraps via DataParallel when
--num_gpus > 1 and that many devices are visible. To restrict the visible
set, export CUDA_VISIBLE_DEVICES before running (setup.sh does this).

Stand-alone example:

    python scripts/train.py \
        --dither_mode sd --lmbda 0.013 --lmbda_p 0.5 \
        --epochs 25 --batch_size 16 --patch_size 256 \
        --data_root path/to/div2k --test_root path/to/kodak \
        --out_dir runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from LTC.dithered_models import Cheng2020RDP  # noqa: E402
from LTC.nested_lattice import NestedE8Product  # noqa: E402
from LTC.net_aux import net_aux_optimizer  # noqa: E402
from LTC.perception import RDPLoss  # noqa: E402

from scripts.data import (  # noqa: E402
    FlatImageFolder,
    autodetect_dataset,
    short_dataset_summary,
)


# --------------------------------------------------------------------- #
# Optional wandb                                                        #
# --------------------------------------------------------------------- #


class _NoOpWandb:
    def init(self, *a, **k): pass
    def log(self, *a, **k): pass
    class config:
        @staticmethod
        def update(*a, **k): pass


def _maybe_wandb():
    if os.environ.get("WANDB_DISABLED", "true").lower() in ("1", "true", "yes"):
        return _NoOpWandb()
    try:
        import wandb
        return wandb
    except Exception as e:
        print(f"[wandb] disabled ({e})")
        return _NoOpWandb()


# --------------------------------------------------------------------- #
# Wrappers                                                              #
# --------------------------------------------------------------------- #


class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0; self.n = 0
    def update(self, v, n=1):
        v = v.item() if torch.is_tensor(v) else float(v)
        self.sum += v * n; self.n += n
    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def psnr(mse: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


# --------------------------------------------------------------------- #
# Training / eval loops                                                 #
# --------------------------------------------------------------------- #


def train_one_epoch(net, criterion, loader, opt, aux_opt, epoch, clip,
                    log_every, log_fn, tb: SummaryWriter, global_step: list):
    net.train()
    device = next(net.parameters()).device
    t0 = time.time()
    meters = {k: AverageMeter() for k in ("loss", "bpp", "mse", "perc", "aux")}
    n_steps = len(loader)
    for i, d in enumerate(loader):
        d = d.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        aux_opt.zero_grad(set_to_none=True)

        out = net(d)
        out = criterion(out, d)
        out["loss"].backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()

        aux = net.aux_loss()
        aux.backward()
        aux_opt.step()

        meters["loss"].update(out["loss"])
        meters["bpp"].update(out["bpp_loss"])
        meters["mse"].update(out["mse_loss"])
        meters["perc"].update(out["perc_loss"])
        meters["aux"].update(aux)

        gs = global_step[0]
        tb.add_scalar("train/loss", out["loss"].item(), gs)
        tb.add_scalar("train/bpp", out["bpp_loss"].item(), gs)
        tb.add_scalar("train/mse", out["mse_loss"].item(), gs)
        tb.add_scalar("train/psnr", float(psnr(out["mse_loss"])), gs)
        tb.add_scalar("train/perc", out["perc_loss"].item(), gs)
        tb.add_scalar("train/aux", float(aux), gs)
        tb.add_scalar("train/lr", opt.param_groups[0]["lr"], gs)
        global_step[0] = gs + 1

        if (i + 1) % log_every == 0 or (i + 1) == n_steps:
            dt = time.time() - t0
            log_fn({
                "phase": "train",
                "epoch": epoch,
                "step": i + 1,
                "of": n_steps,
                "loss": meters["loss"].avg,
                "bpp": meters["bpp"].avg,
                "mse": meters["mse"].avg,
                "psnr": float(psnr(torch.tensor(meters["mse"].avg))),
                "perc": meters["perc"].avg,
                "aux": meters["aux"].avg,
                "time_s": dt,
                "img_per_s": ((i + 1) * loader.batch_size) / max(dt, 1e-6),
            })
    return meters


@torch.no_grad()
def eval_loop(net, criterion, loader, epoch, log_fn, tb: SummaryWriter,
              n_image_samples: int = 4):
    net.eval()
    device = next(net.parameters()).device
    meters = {k: AverageMeter() for k in ("loss", "bpp", "mse", "perc", "aux")}
    sample_in, sample_rec = None, None
    for bidx, d in enumerate(loader):
        d = d.to(device, non_blocking=True)
        out = net(d)
        loss_dict = criterion(out, d)
        meters["loss"].update(loss_dict["loss"])
        meters["bpp"].update(loss_dict["bpp_loss"])
        meters["mse"].update(loss_dict["mse_loss"])
        meters["perc"].update(loss_dict["perc_loss"])
        meters["aux"].update(net.aux_loss())
        if bidx == 0:
            k = min(n_image_samples, d.shape[0])
            sample_in = d[:k].detach().cpu().clamp(0, 1)
            sample_rec = out["x_hat"][:k].detach().cpu().clamp(0, 1)
    tb.add_scalar("test/loss", meters["loss"].avg, epoch)
    tb.add_scalar("test/bpp", meters["bpp"].avg, epoch)
    tb.add_scalar("test/mse", meters["mse"].avg, epoch)
    tb.add_scalar("test/psnr", float(psnr(torch.tensor(meters["mse"].avg))), epoch)
    tb.add_scalar("test/perc", meters["perc"].avg, epoch)
    tb.add_scalar("test/aux", meters["aux"].avg, epoch)
    if sample_in is not None:
        tb.add_image("test/input", make_grid(sample_in, nrow=2), epoch)
        tb.add_image("test/recon", make_grid(sample_rec, nrow=2), epoch)
        residual = (sample_in - sample_rec).abs()
        residual = residual / residual.amax().clamp_min(1e-6)
        tb.add_image("test/residual_abs_norm", make_grid(residual, nrow=2), epoch)
    log_fn({
        "phase": "test",
        "epoch": epoch,
        "loss": meters["loss"].avg,
        "bpp": meters["bpp"].avg,
        "mse": meters["mse"].avg,
        "psnr": float(psnr(torch.tensor(meters["mse"].avg))),
        "perc": meters["perc"].avg,
        "aux": meters["aux"].avg,
    })
    return meters["loss"].avg


# --------------------------------------------------------------------- #
# main                                                                  #
# --------------------------------------------------------------------- #


def _env_int(key: str, default):
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def parse_args(argv=None):
    p = argparse.ArgumentParser("WALLOC train")
    # data
    p.add_argument("--data_root", default=os.environ.get("WALLOC_TRAIN_INPUT"),
                   help="train image folder (default $WALLOC_TRAIN_INPUT; "
                        "autodetect if unset)")
    p.add_argument("--test_root", default=os.environ.get("WALLOC_TEST_INPUT"),
                   help="held-out images for per-epoch eval "
                        "(default $WALLOC_TEST_INPUT)")
    p.add_argument("--min_train_size", type=int, default=320,
                   help="drop train images smaller than this on the short side")
    p.add_argument("--max_train_images", type=int, default=None,
                   help="cap train set (debugging)")
    # arch
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--lattice_name", default="E8Product")
    p.add_argument("--N_integral", type=int, default=512,
                   help="MC samples for likelihood (512 is a good tradeoff)")
    # RDP
    p.add_argument("--lmbda", type=float, default=0.013)
    p.add_argument("--lmbda_p", type=float, default=0.0)
    p.add_argument("--perc", choices=("sw", "sinkhorn"), default="sw")
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--n_proj", type=int, default=128)
    p.add_argument("--max_patches", type=int, default=4096)
    # dither
    p.add_argument("--dither_mode", choices=("det", "pd", "sd", "qsd"), default="sd")
    p.add_argument("--s_dither", type=float, default=1.0)
    p.add_argument("--qsd_a", type=int, default=2)
    # opt
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=16,
                   help="total batch across all GPUs")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--aux_lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--num_gpus", type=int,
                   default=_env_int("WALLOC_NUM_GPUS", 0),
                   help="0 = auto-use all visible; 1 = single GPU (no DP); "
                        ">1 = DataParallel across first N visible")
    # NTC init speeds convergence dramatically
    p.add_argument("--load_ntc", action="store_true", default=True)
    p.add_argument("--ntc_quality", type=int, default=4)
    # io
    p.add_argument("--out_dir",
                   default=os.environ.get("WALLOC_OUTPUT_DIR", "runs"),
                   help="output root (default $WALLOC_OUTPUT_DIR or ./runs)")
    p.add_argument("--tag", default=None, help="run name (auto if omitted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None, help="resume path")
    return p.parse_args(argv)


def make_tag(args) -> str:
    parts = [f"{args.dither_mode}", f"l{args.lmbda}", f"lp{args.lmbda_p}"]
    if args.dither_mode in ("pd", "qsd"):
        parts.append(f"s{args.s_dither}")
    if args.dither_mode == "qsd":
        parts.append(f"a{args.qsd_a}")
    return "_".join(parts)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    args.data_root = autodetect_dataset(args.data_root, purpose="train")
    if args.test_root is None:
        try:
            args.test_root = autodetect_dataset(None, purpose="test")
        except FileNotFoundError:
            print("[warn] no test set found — using a held-out slice of train.")
            args.test_root = args.data_root

    tag = args.tag or make_tag(args)
    out_root = Path(args.out_dir) / tag
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "metrics.jsonl"
    ck_path = out_root / "model.pt"
    summary_path = out_root / "summary.json"
    tb_dir = out_root / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(tb_dir))
    tb.add_text("args", "\n".join(f"{k}={v}" for k, v in sorted(vars(args).items())), 0)
    print(f"[tb] writing to {tb_dir}")

    wandb = _maybe_wandb()
    wandb.init(project="walloc", name=tag)
    wandb.config.update(vars(args))

    # ----- dataset ------------------------------------------------------
    train_tf = transforms.Compose([
        transforms.RandomCrop(args.patch_size, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.Compose([
        transforms.CenterCrop(args.patch_size),
        transforms.ToTensor(),
    ])
    train_ds = FlatImageFolder(
        args.data_root, transform=train_tf,
        min_size=args.min_train_size, max_images=args.max_train_images,
    )
    test_ds = FlatImageFolder(args.test_root, transform=test_tf)

    n_train, (mw, mh) = short_dataset_summary(args.data_root)
    n_test, _ = short_dataset_summary(args.test_root)
    print(f"[data] train={args.data_root} (~{n_train} imgs, median {mw}x{mh}) "
          f"used={len(train_ds)} | test={args.test_root} (~{n_test} imgs)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(2, args.batch_size // 2),
        num_workers=max(1, args.num_workers // 2),
        shuffle=False, pin_memory=True,
    )

    # ----- model --------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = "cuda"
    visible = torch.cuda.device_count()
    n_gpus = args.num_gpus if args.num_gpus > 0 else visible
    n_gpus = min(n_gpus, visible)
    print(f"[gpu] visible={visible} use={n_gpus}")

    nested = None
    if args.dither_mode == "qsd":
        nested = NestedE8Product(channels=args.channels, a=args.qsd_a)
        print(f"[qsd] a={args.qsd_a} R_c={nested.R_c_bits_per_dim:.4f} bits/dim")

    net = Cheng2020RDP(
        N=args.channels, N_integral=args.N_integral,
        lattice_name=args.lattice_name,
        dither_mode=args.dither_mode, s_dither=args.s_dither,
        nested_quantizer=nested,
    )

    if args.load_ntc:
        from compressai.zoo import image_models as pretrained_models
        print(f"[init] loading NTC cheng2020-attn q={args.ntc_quality}")
        ntc = pretrained_models["cheng2020-attn"](
            quality=args.ntc_quality, metric="mse", pretrained=True, progress=False
        )
        skip = {
            "gaussian_conditional._offset",
            "gaussian_conditional._quantized_cdf",
            "gaussian_conditional._cdf_length",
            "gaussian_conditional.scale_table",
        }
        sd = {k: v for k, v in ntc.state_dict().items() if k not in skip}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}")

    net = net.to(device)
    opts = net_aux_optimizer(
        net,
        {
            "net": {"type": "Adam", "lr": args.lr},
            "aux": {"type": "Adam", "lr": args.aux_lr},
        },
    )
    optimizer, aux_opt = opts["net"], opts["aux"]
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = RDPLoss(
        lmbda=args.lmbda, lmbda_p=args.lmbda_p, perc=args.perc,
        patch=args.patch, n_proj=args.n_proj, max_patches=args.max_patches,
    )

    last_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        ck = torch.load(args.checkpoint, map_location=device)
        last_epoch = ck["epoch"] + 1
        net.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        aux_opt.load_state_dict(ck["aux_optimizer"])
        lr_scheduler.load_state_dict(ck["lr_scheduler"])
        print(f"[resume] from epoch {last_epoch} ({args.checkpoint})")

    if n_gpus > 1:
        print(f"[dp] wrapping in DataParallel across {n_gpus} GPUs")
        net = CustomDataParallel(net, device_ids=list(range(n_gpus)))

    # ----- log file -----------------------------------------------------
    log_file = open(log_path, "a", buffering=1)
    def log_fn(rec):
        rec["tag"] = tag
        rec["time"] = time.time()
        log_file.write(json.dumps(rec) + "\n")
        wandb.log(rec)
        if rec["phase"] == "train" and rec["step"] == rec["of"]:
            print(
                f"  ep{rec['epoch']:02d} step{rec['step']:>4}/{rec['of']} "
                f"loss={rec['loss']:.3f} bpp={rec['bpp']:.4f} "
                f"psnr={rec['psnr']:.2f} perc={rec['perc']:.3e} "
                f"ips={rec['img_per_s']:.1f}"
            )
        if rec["phase"] == "test":
            print(
                f"  ep{rec['epoch']:02d} TEST loss={rec['loss']:.3f} "
                f"bpp={rec['bpp']:.4f} psnr={rec['psnr']:.2f} "
                f"perc={rec['perc']:.3e}"
            )

    # ----- train loop ---------------------------------------------------
    log_every = max(1, len(train_loader) // 4)
    best_loss = float("inf")
    history = []
    global_step = [last_epoch * len(train_loader)]
    for epoch in range(last_epoch, args.epochs):
        print(f"\n[epoch {epoch}] lr={optimizer.param_groups[0]['lr']:.2e}")
        train_one_epoch(
            net, criterion, train_loader, optimizer, aux_opt, epoch,
            args.clip_max_norm, log_every, log_fn, tb, global_step,
        )
        loss = eval_loop(net, criterion, test_loader, epoch, log_fn, tb)
        lr_scheduler.step(loss)
        history.append(float(loss))

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        state_dict = (
            net.module.state_dict() if isinstance(net, CustomDataParallel)
            else net.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "state_dict": state_dict,
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_opt.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": vars(args),
            },
            ck_path,
        )
        if is_best:
            torch.save({"state_dict": state_dict, "args": vars(args), "epoch": epoch},
                       out_root / "model_best.pt")

    summary = {
        "tag": tag,
        "args": vars(args),
        "best_test_loss": best_loss,
        "history": history,
        "checkpoint": str(ck_path),
        "best_checkpoint": str(out_root / "model_best.pt"),
        "log": str(log_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_file.close()
    tb.flush()
    tb.close()
    print(f"[done] {tag} best_test_loss={best_loss:.4f} -> {out_root}")
    return summary


if __name__ == "__main__":
    main()
