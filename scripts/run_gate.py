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

------------------------------------------------------------------------
PATCH NOTES (round-4 blocker fixes, addresses B1-B4 from review):

B1 (PD training instability). Round-3 pd_lp_hi ran 12 epochs without
   the lr scheduler ever firing — ReduceLROnPlateau couldn't see a
   plateau through the noisy perception loss. The default schedule is
   now cosine+warmup (deterministic, fires unconditionally). Plumbed
   through train.py via --lr_schedule.

B2 (bpp matching). Round-3 spread was 15% (target 10%). Pulling det
   lmbda further down: 0.031 * 0.504/0.547 ~ 0.028. SD anchor 0.013
   stays, PD 0.018 stays (its endpoints 0.518 / 0.476 average 0.497 —
   already on anchor, range is intrinsic perception-bpp dynamics).

B3 (single-seed → 3 seeds). --seeds 0,1,2 expands the matrix; tags get
   _seed{N} suffix (seed 0 keeps the bare tag for back-compat with
   existing checkpoints). aggregate() groups by base tag and reports
   mean ± std. decide_gate consumes the aggregated rows so criteria
   are evaluated on seed-averaged metrics.

B4 (no external baseline). scripts/baseline_cheng.py runs vanilla
   cheng2020-attn (q=1..6) on Kodak as a reference R-D curve. HiFiC
   needs a separate codebase — left as a TODO in baseline_cheng.py.

History:
  Round-1: shared lmbda=0.013 -> bpp 0.50/0.40/0.34, P1/P3 invalid.
  Round-2: lmbda_pd=0.020, lmbda_det=0.038 -> bpp 0.50/0.55/0.62,
           perc_scale=255**2 fix; PD P2 fail at "matched" pressure.
  Round-3: lmbda_pd=0.018, lmbda_det=0.031, lp {0,0.05,0.2} -> bpp
           spread 15%, SD P2 strong (sw2 -54%), PD P2 fail BUT
           pd_lp_hi never converged (B1 problem).
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


def gate_matrix(args) -> List[Dict]:
    """6 critical runs + 1 ablation that decide the gate.

    Per-mode lmbda is essential: at a shared lmbda, PD and det land at
    LOWER bpp than SD; we INCREASE lmbda for PD/det so all three modes
    land near the same bpp. Calibrate from the first round-2 run if the
    bpp spread is still > 10%.

    Per-mode lmbda_p: the grid {lp0, lp_lo, lp_hi} is shared across modes
    so the perception ramp is comparable.
    """
    return [
        # SD anchor + perception ramp
        {"tag": "sd_lp0",        "mode": "sd",  "lp": args.lp_zero, "s": 1.0,  "lmbda": args.lmbda_sd},
        {"tag": "sd_lp_lo",      "mode": "sd",  "lp": args.lp_lo,   "s": 1.0,  "lmbda": args.lmbda_sd},
        {"tag": "sd_lp_hi",      "mode": "sd",  "lp": args.lp_hi,   "s": 1.0,  "lmbda": args.lmbda_sd},
        # PD anchor + perception ramp (lmbda raised to match SD bpp)
        {"tag": "pd_lp0_s125",   "mode": "pd",  "lp": args.lp_zero, "s": 1.25, "lmbda": args.lmbda_pd},
        {"tag": "pd_lp_lo_s125", "mode": "pd",  "lp": args.lp_lo,   "s": 1.25, "lmbda": args.lmbda_pd},
        {"tag": "pd_lp_hi_s125", "mode": "pd",  "lp": args.lp_hi,   "s": 1.25, "lmbda": args.lmbda_pd},
        # Ablation: deterministic cannot reach the SD/PD perception floor.
        {"tag": "det_lp_hi",     "mode": "det", "lp": args.lp_hi,   "s": 1.0,  "lmbda": args.lmbda_det},
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
        "--lmbda", str(cfg["lmbda"]),
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
        "--lr_schedule", args.lr_schedule,
        "--seed", str(cfg.get("seed", 0)),
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


def _strip_seed(tag: str) -> str:
    # "sd_lp0_seed1" -> "sd_lp0"; "sd_lp0" -> "sd_lp0".
    import re
    return re.sub(r"_seed\d+$", "", tag)


def aggregate(eval_jsons: List[Path]) -> Dict:
    """Read per-run eval JSONs; group by base tag (B3) and average over seeds.

    Per-image rows are kept as raw; the gate consumes the averaged table.
    """
    import statistics

    per_seed = []
    for j in eval_jsons:
        if not j.exists():
            continue
        with open(j) as f:
            d = json.load(f)
        row = {"tag": d["tag"], "base_tag": _strip_seed(d["tag"])}
        row.update(d["summary"])
        ta = d["train_args"]
        row["mode"] = ta["dither_mode"]
        row["lmbda"] = ta["lmbda"]
        row["lmbda_p"] = ta["lmbda_p"]
        row["s_dither"] = ta["s_dither"]
        row["seed"] = ta.get("seed", 0)
        per_seed.append(row)

    # Group by base_tag, average metrics, record n_seeds + std for the
    # key columns. metric_keys is the union of numeric summary keys; if a
    # key isn't present in every seed (e.g. lpips_mean), it's averaged
    # over the seeds that have it.
    by_base: Dict[str, List[Dict]] = {}
    for r in per_seed:
        by_base.setdefault(r["base_tag"], []).append(r)

    metric_keys = (
        "bpp_mean", "psnr_mean", "mse_mean", "sw2_patch_mean", "lpips_mean",
    )
    table = []
    for base, rows in by_base.items():
        ref = rows[0]
        agg = {
            "tag": base,
            "n_seeds": len(rows),
            "seeds": sorted(r["seed"] for r in rows),
            "mode": ref["mode"],
            "lmbda": ref["lmbda"],
            "lmbda_p": ref["lmbda_p"],
            "s_dither": ref["s_dither"],
        }
        for k in metric_keys:
            vals = [r[k] for r in rows if k in r]
            if not vals:
                continue
            agg[k] = sum(vals) / len(vals)
            if len(vals) >= 2:
                agg[f"{k}_std"] = statistics.stdev(vals)
        table.append(agg)
    return {"table": table, "per_seed": per_seed}


def decide_gate(table: List[Dict]) -> Dict:
    """Apply the kill criteria from the planning doc (Phần 4.5).

    P1: SD beats PD on perception (SW_2) at matched lambda_p, for >=2 lp values.
    P2: SW_2 monotone non-increasing as lambda_p grows (per mode).
    P3: deterministic with high lambda_p cannot reach SD's perception floor.

    All three criteria are CONDITIONAL on bpp matching within 10%; if that
    sanity check fails, the SW_2 comparison is comparing apples to oranges
    and the gate verdict is marked invalid.
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

    # Find the lp values actually used (after the perc_scale fix the grid
    # is no longer fixed to {0.5, 2.0}; introspect from the table).
    lp_values = sorted({r["lmbda_p"] for r in table if r["lmbda_p"] > 0})
    p1_count = 0
    for lp in lp_values:
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

    # P3 compares det at the highest lp to SD at the highest lp.
    lp_hi = max(lp_values) if lp_values else None
    det = get("det", lp_hi) if lp_hi is not None else None
    sd_hi = get("sd", lp_hi) if lp_hi is not None else None
    if det is not None and sd_hi is not None:
        results["P3_det_cannot_reach_sd_perception"] = (
            det["sw2_patch_mean"] > sd_hi["sw2_patch_mean"] * 1.05
        )
        notes.append(
            f"P3 (lp={lp_hi}): det sw2={det['sw2_patch_mean']:.3e} "
            f"vs sd sw2={sd_hi['sw2_patch_mean']:.3e}"
        )
    else:
        results["P3_det_cannot_reach_sd_perception"] = None

    # bpp-matched sanity check: SW2 comparison is invalid if rates spread
    # by more than ~10%. If this fails, P1 and P3 are NOT trustworthy.
    bpps = [r["bpp_mean"] for r in table]
    if bpps:
        bpp_spread = (max(bpps) - min(bpps)) / max(min(bpps), 1e-9)
        bpp_ok = bpp_spread < 0.10
        results["bpp_matched_within_10pct"] = bpp_ok
        notes.append(
            f"bpp spread: min={min(bpps):.4f} max={max(bpps):.4f} "
            f"rel={bpp_spread:.1%} -> "
            f"{'OK' if bpp_ok else 'INVALID (retune lmbda_pd / lmbda_det)'}"
        )
        if not bpp_ok:
            # If bpp matching failed, demote P1/P3 to None (invalid) so we
            # don't draw conclusions from confounded comparisons.
            results["P1_sd_beats_pd_on_perception"] = None
            results["P3_det_cannot_reach_sd_perception"] = None
            notes.append(
                "P1 and P3 marked invalid: SW_2 comparison requires "
                "matched bpp. Retune per-mode lmbda and rerun."
            )

    # GATE_PASS requires every non-None check to be True. None counts as
    # "indeterminate", neither passing nor failing the gate.
    non_none = [v for v in results.values() if v is not None and isinstance(v, bool)]
    if not non_none:
        results["GATE_PASS"] = None
    else:
        results["GATE_PASS"] = all(non_none) and (
            results["bpp_matched_within_10pct"] is True
        )
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
    # ---- training knobs (passed through to scripts/train.py) ----
    # Per-mode lmbda: PD and det land at LOWER bpp than SD at the same lmbda,
    # so we RAISE lmbda for them to match SD's ~0.50 bpp anchor. (The
    # round-1 calibration went the wrong direction; see PATCH NOTES at the
    # top of this file.) These defaults are first-order estimates; verify
    # with the first round-2 run and adjust if bpp_spread > 10%.
    p.add_argument("--lmbda",     type=float, default=None,
                   help="if set, overrides per-mode lmbdas (back-compat; "
                        "almost always wrong because bpp won't match)")
    p.add_argument("--lmbda_sd",  type=float, default=0.013,
                   help="SD anchor; rounds 1+2 both landed bpp ~ 0.50")
    p.add_argument("--lmbda_pd",  type=float, default=0.018,
                   help="PD; round-2 used 0.020 -> bpp 0.55 (7% over); "
                        "scaled to 0.018 to land bpp ~ 0.50")
    p.add_argument("--lmbda_det", type=float, default=0.028,
                   help="det; round-3 used 0.031 -> bpp 0.547 (9% over); "
                        "scaled to 0.028 to land bpp ~ 0.50")
    # ---- B1: lr schedule (passed through to train.py) ----
    p.add_argument("--lr_schedule", choices=("cosine", "plateau"),
                   default="cosine",
                   help="cosine: deterministic warmup+cosine decay (fixes "
                        "the round-3 pd_lp_hi non-convergence); "
                        "plateau: legacy ReduceLROnPlateau")
    # ---- B3: multi-seed ----
    p.add_argument("--seeds", default="0",
                   help="comma-separated seeds; each config in the matrix "
                        "is run once per seed. Tags get _seed{N} suffix "
                        "for seed != 0 (seed 0 keeps the bare tag).")
    # ---- perception ramp ----
    # New defaults assume the perc_scale=255**2 fix in LTC/perception.py.
    # Old grid {0, 0.5, 2.0} is now ~4000x too weak. The new grid puts
    # perc contribution at ~30% / ~150% of MSE term at lp_lo / lp_hi.
    p.add_argument("--lp_zero", type=float, default=0.0)
    p.add_argument("--lp_lo",   type=float, default=0.05)
    p.add_argument("--lp_hi",   type=float, default=0.2)
    # ---- other training knobs ----
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=192)
    p.add_argument("--N_integral", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ntc_quality", type=int, default=4,
                   help="cheng2020-attn pretrained checkpoint; "
                        "q=4 (N=192) matches --channels 192 default")
    p.add_argument("--num_gpus", type=int,
                   default=_env_int("WALLOC_NUM_GPUS", 0))
    # ---- orchestration ----
    p.add_argument("--runs", default=None,
                   help="comma-separated tags to include (default: all)")
    p.add_argument("--force_retrain", action="store_true")
    p.add_argument("--force_eval", action="store_true")
    p.add_argument("--skip_train", action="store_true",
                   help="only run eval on existing checkpoints")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Back-compat: --lmbda VALUE applies to all three modes (the old behavior
    # that produced the bpp-mismatch problem; kept only for legacy invocation).
    if args.lmbda is not None:
        print(f"[warn] --lmbda={args.lmbda} overrides per-mode lmbdas; "
              f"bpp matching will likely fail.")
        args.lmbda_sd = args.lmbda_pd = args.lmbda_det = args.lmbda
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.data_root is None:
        args.data_root = autodetect_dataset(None, purpose="train")
    if args.test_root is None:
        try:
            args.test_root = autodetect_dataset(None, purpose="test")
        except FileNotFoundError:
            print("[warn] no test set autodetected; eval will fail if not provided")

    base_matrix = gate_matrix(args)
    if args.runs:
        wanted = set(s.strip() for s in args.runs.split(","))
        base_matrix = [c for c in base_matrix if c["tag"] in wanted]

    # B3: expand each base config across seeds. seed 0 keeps the bare tag
    # (back-compat with round-1..3 checkpoints); other seeds get _seedN.
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    matrix = []
    for cfg in base_matrix:
        for s in seeds:
            row = dict(cfg)
            row["seed"] = s
            row["base_tag"] = cfg["tag"]
            row["tag"] = cfg["tag"] if s == 0 else f"{cfg['tag']}_seed{s}"
            matrix.append(row)

    print(f"[gate] {len(matrix)} runs planned across {len(seeds)} seed(s): "
          f"{[c['tag'] for c in matrix]}")
    print(f"[gate] lmbda: sd={args.lmbda_sd} pd={args.lmbda_pd} det={args.lmbda_det}")
    print(f"[gate] lp grid: {args.lp_zero} / {args.lp_lo} / {args.lp_hi} "
          f"(assumes perc_scale=255**2 in RDPLoss)")
    print(f"[gate] lr_schedule={args.lr_schedule} seeds={seeds}")
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
    multi_seed = any(r.get("n_seeds", 1) > 1 for r in agg["table"])
    hdr = f"{'tag':<18}{'mode':<5}{'lp':>7}{'s':>6}{'n':>3}  {'bpp':>7}{'psnr':>8}{'sw2':>11}"
    if any("lpips_mean" in r for r in agg["table"]):
        hdr += f"{'lpips':>9}"
    print(hdr)
    for r in agg["table"]:
        n = r.get("n_seeds", 1)
        line = (
            f"{r['tag']:<18}{r['mode']:<5}{r['lmbda_p']:>7.3f}{r['s_dither']:>6.2f}"
            f"{n:>3}  {r['bpp_mean']:>7.4f}{r['psnr_mean']:>8.3f}"
            f"{r['sw2_patch_mean']:>11.3e}"
        )
        if "lpips_mean" in r:
            line += f"{r['lpips_mean']:>9.4f}"
        print(line)
        if multi_seed and "sw2_patch_mean_std" in r:
            print(f"{'  +/- std':<18}{'':<5}{'':<7}{'':<6}{'':<3}  "
                  f"{r.get('bpp_mean_std', 0):>7.4f}"
                  f"{r.get('psnr_mean_std', 0):>8.3f}"
                  f"{r['sw2_patch_mean_std']:>11.3e}")

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