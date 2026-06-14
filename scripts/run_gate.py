"""
Run the WALLOC Phase-1 gate matrix end-to-end.

What this does (one session):
  1. Train a set of configs sequentially (PD vs SD across several lambda_p
     values, plus a deterministic baseline ablation).
  2. Evaluate each best checkpoint on the test set.
  3. Aggregate all eval JSONs into one comparison table.
  4. Apply pass/kill criteria from the WALLOC planning doc (Phần 4.5).

Resume after timeout: re-run with the same args — finished runs are
skipped (summary.json + eval_kodak.json are the markers).

Picks up env defaults from setup.sh: WALLOC_OUTPUT_DIR, WALLOC_TRAIN_INPUT,
WALLOC_TEST_INPUT, WALLOC_NUM_GPUS.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.data import autodetect_dataset  # noqa: E402


# --------------------------------------------------------------------- #
# Gate matrix                                                           #
# --------------------------------------------------------------------- #


def gate_matrix(lmbda: float) -> List[Dict]:
    """6 critical runs + 1 ablation that decide the gate."""
    return [
        # baselines (lambda_p = 0)
        {"tag": "sd_lp0",        "mode": "sd",  "lp": 0.0, "s": 1.0},
        {"tag": "pd_lp0_s125",   "mode": "pd",  "lp": 0.0, "s": 1.25},
        # perception ramp
        {"tag": "sd_lp_lo",      "mode": "sd",  "lp": 0.5, "s": 1.0},
        {"tag": "sd_lp_hi",      "mode": "sd",  "lp": 2.0, "s": 1.0},
        {"tag": "pd_lp_lo_s125", "mode": "pd",  "lp": 0.5, "s": 1.25},
        {"tag": "pd_lp_hi_s125", "mode": "pd",  "lp": 2.0, "s": 1.25},
        # ablation: deterministic cannot reach the perception floor SD/PD can.
        {"tag": "det_lp_hi",     "mode": "det", "lp": 2.0, "s": 1.0},
    ]


# --------------------------------------------------------------------- #
# Subprocess wrappers                                                   #
# --------------------------------------------------------------------- #


def run_subprocess(cmd: List[str], log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}\n  log: {log_path}")
    t0 = time.time()
    with open(log_path, "ab") as f:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line.decode(errors="replace"))
            sys.stdout.flush()
            f.write(line)
        proc.wait()
    print(f"  exit={proc.returncode} time={time.time() - t0:.1f}s")
    return proc.returncode


def train_one(cfg, args, out_root: Path) -> Path:
    tag = cfg["tag"]
    run_dir = out_root / tag
    summary = run_dir / "summary.json"
    if summary.exists() and not args.force_retrain:
        print(f"[skip] {tag}: summary.json exists")
        return run_dir / "model_best.pt"

    cmd = [
        sys.executable, "-u", "scripts/train.py",
        "--dither_mode", cfg["mode"],
        "--lmbda", str(args.lmbda),
        "--lmbda_p", str(cfg["lp"]),
        "--s_dither", str(cfg["s"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--patch_size", str(args.patch_size),
        "--channels", str(args.channels),
        "--N_integral", str(args.N_integral),
        "--num_workers", str(args.num_workers),
        "--out_dir", str(out_root),
        "--tag", tag,
        "--ntc_quality", str(args.ntc_quality),
        "--num_gpus", str(args.num_gpus),
    ]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    if args.test_root:
        cmd += ["--test_root", args.test_root]
    rc = run_subprocess(cmd, run_dir / "train.log")
    if rc != 0:
        raise RuntimeError(f"train failed for {tag} (rc={rc})")
    return run_dir / "model_best.pt"


def eval_one(ckpt: Path, args, out_root: Path) -> Path:
    out_json = out_root / ckpt.parent.name / "eval_kodak.json"
    if out_json.exists() and not args.force_eval:
        print(f"[skip-eval] {out_json.name} exists")
        return out_json
    tb_dir = out_root / ckpt.parent.name / "tb_eval"
    cmd = [
        sys.executable, "-u", "scripts/eval.py",
        "--checkpoint", str(ckpt),
        "--out_json", str(out_json),
        "--tb_dir", str(tb_dir),
    ]
    if args.test_root:
        cmd += ["--test_root", args.test_root]
    run_subprocess(cmd, out_json.parent / "eval.log")
    return out_json


# --------------------------------------------------------------------- #
# Aggregation + gate decision                                           #
# --------------------------------------------------------------------- #


def aggregate(eval_jsons: List[Path]) -> Dict:
    table = []
    for j in eval_jsons:
        if not j.exists():
            continue
        with open(j) as f:
            d = json.load(f)
        row = {"tag": d["tag"]}
        row.update(d["summary"])
        ta = d["train_args"]
        row["mode"] = ta["dither_mode"]
        row["lmbda"] = ta["lmbda"]
        row["lmbda_p"] = ta["lmbda_p"]
        row["s_dither"] = ta["s_dither"]
        table.append(row)
    return {"table": table}


def decide_gate(table: List[Dict]) -> Dict:
    """Apply the kill criteria from the planning doc (Phần 4.5).

    P1: SD beats PD on perception (SW_2) at matched lambda_p, for >=2 lp values.
    P2: SW_2 monotone non-increasing as lambda_p grows (per mode).
    P3: deterministic with high lambda_p cannot reach SD's perception floor.
    """
    def get(mode, lp):
        for r in table:
            if r["mode"] == mode and abs(r["lmbda_p"] - lp) < 1e-9:
                return r
        return None

    results = {}
    notes = []

    for mode in ("sd", "pd"):
        rows = sorted(
            [r for r in table if r["mode"] == mode],
            key=lambda r: r["lmbda_p"],
        )
        sws = [r["sw2_patch_mean"] for r in rows]
        ok = all(sws[i + 1] <= sws[i] * 1.05 for i in range(len(sws) - 1))
        results[f"P2_{mode}_monotone_in_lp"] = ok
        notes.append(f"{mode} sw2 vs lp: {sws}")

    p1_count = 0
    for lp in (0.5, 2.0):
        sd = get("sd", lp); pd = get("pd", lp)
        if sd is None or pd is None:
            continue
        if sd["sw2_patch_mean"] < pd["sw2_patch_mean"]:
            p1_count += 1
        notes.append(
            f"lp={lp}: SD sw2={sd['sw2_patch_mean']:.3e} "
            f"PD sw2={pd['sw2_patch_mean']:.3e}"
        )
    results["P1_sd_beats_pd_on_perception"] = p1_count >= 2

    det = get("det", 2.0); sd_hi = get("sd", 2.0)
    if det is not None and sd_hi is not None:
        results["P3_det_cannot_reach_sd_perception"] = (
            det["sw2_patch_mean"] > sd_hi["sw2_patch_mean"] * 1.05
        )
        notes.append(
            f"P3: det sw2={det['sw2_patch_mean']:.3e} "
            f"vs sd sw2={sd_hi['sw2_patch_mean']:.3e}"
        )
    else:
        results["P3_det_cannot_reach_sd_perception"] = None

    results["GATE_PASS"] = all(v for v in results.values() if v is not None)
    results["notes"] = notes
    return results


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
    p = argparse.ArgumentParser("WALLOC gate matrix")
    p.add_argument("--out_root",
                   default=os.environ.get("WALLOC_OUTPUT_DIR", "runs") + "_gate"
                   if "WALLOC_OUTPUT_DIR" in os.environ
                   else "runs_gate")
    p.add_argument("--data_root", default=os.environ.get("WALLOC_TRAIN_INPUT"))
    p.add_argument("--test_root", default=os.environ.get("WALLOC_TEST_INPUT"))
    # training knobs (passed through to scripts/train.py)
    p.add_argument("--lmbda", type=float, default=0.013)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--N_integral", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ntc_quality", type=int, default=4)
    p.add_argument("--num_gpus", type=int,
                   default=_env_int("WALLOC_NUM_GPUS", 0))
    # orchestration
    p.add_argument("--runs", default=None,
                   help="comma-separated tags to include (default: all)")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--force_eval", action="store_true")
    p.add_argument("--skip_train", action="store_true",
                   help="only run eval on existing checkpoints")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.data_root is None:
        args.data_root = autodetect_dataset(None, purpose="train")
    if args.test_root is None:
        try:
            args.test_root = autodetect_dataset(None, purpose="test")
        except FileNotFoundError:
            print("[warn] no test set autodetected; eval will fail if not provided")

    matrix = gate_matrix(args.lmbda)
    if args.runs:
        wanted = set(s.strip() for s in args.runs.split(","))
        matrix = [c for c in matrix if c["tag"] in wanted]
    print(f"[gate] {len(matrix)} runs planned: {[c['tag'] for c in matrix]}")
    print(f"[gate] out_root={out_root}")
    print(f"[gate] data_root={args.data_root}")
    print(f"[gate] test_root={args.test_root}")

    eval_jsons = []
    for cfg in matrix:
        try:
            if args.skip_train:
                ckpt = out_root / cfg["tag"] / "model_best.pt"
            else:
                ckpt = train_one(cfg, args, out_root)
            j = eval_one(ckpt, args, out_root)
            eval_jsons.append(j)
        except Exception as e:
            print(f"[error] {cfg['tag']}: {e}")

    agg = aggregate(eval_jsons)
    gate = decide_gate(agg["table"]) if agg["table"] else {"GATE_PASS": None,
                                                            "notes": ["no eval rows"]}
    out = {"matrix": matrix, "table": agg["table"], "gate": gate}
    final = out_root / "GATE_RESULTS.json"
    with open(final, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 72)
    print("GATE TABLE (lower sw2 = better perception; higher psnr = better fidelity)")
    print("=" * 72)
    hdr = f"{'tag':<18}{'mode':<5}{'lp':>6}{'s':>6}  {'bpp':>7}{'psnr':>8}{'sw2':>11}"
    if any("lpips_mean" in r for r in agg["table"]):
        hdr += f"{'lpips':>9}"
    print(hdr)
    for r in agg["table"]:
        line = (
            f"{r['tag']:<18}{r['mode']:<5}{r['lmbda_p']:>6.2f}{r['s_dither']:>6.2f}  "
            f"{r['bpp_mean']:>7.4f}{r['psnr_mean']:>8.3f}{r['sw2_patch_mean']:>11.3e}"
        )
        if "lpips_mean" in r:
            line += f"{r['lpips_mean']:>9.4f}"
        print(line)

    print("\n" + "=" * 72)
    print("GATE DECISION")
    print("=" * 72)
    for k, v in gate.items():
        if k == "notes":
            print(f"\nnotes:")
            for n in v:
                print(f"  - {n}")
        else:
            mark = "OK" if v is True else ("FAIL" if v is False else "n/a")
            print(f"  [{mark}] {k}")
    print(f"\n-> {final}")


if __name__ == "__main__":
    main()
