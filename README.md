# WALLOC

Wasserstein-Anchored Lattice Codec — extends the LTC framework (arXiv
2503.17558) with a perception-aware RDP objective on full-resolution images.
Three operational dither modes are implemented and verified against the
crypto lemma + Prop 3.3 + the QSD volume correction:

| Mode | Common randomness | Per-coord side rate |
|------|------------------|--------------------:|
| **PD-LTC**  | private dither           | `R_c = 0`         |
| **SD-LTC**  | shared continuous dither | `R_c = infinity`  |
| **QSD-LTC** | quantized shared dither  | `R_c = log2(a)`   |

`det` is a deterministic (rounding) ablation used to confirm gate
criterion P3.

## Layout

```
walloc/
├── setup.sh               # env bootstrap (GPUs, paths, deps)
├── requirements.txt
├── LTC/                   # model package
│   ├── dithered_models.py # DitheredGaussianConditionalLattice + Cheng2020RDP
│   ├── nested_lattice.py  # NestedE8Product (cosets for QSD)
│   ├── perception.py      # extract_patches, sliced_w2, sinkhorn, RDPLoss
│   ├── models_compressai.py
│   ├── quantizers.py
│   ├── entropy_models.py
│   ├── net_aux.py
│   └── flows/
├── scripts/               # CLI entrypoints
│   ├── data.py            # FlatImageFolder + autodetect_dataset
│   ├── train.py           # single config train  (DataParallel + TB)
│   ├── eval.py            # eval on Kodak/CLIC   (TB previews + JSON)
│   └── run_gate.py        # 7-run gate matrix orchestrator
├── notebooks/
│   └── walloc_kaggle.ipynb
└── tests/
    └── test_walloc.py     # crypto lemma, Prop 3.3, QSD volume correction, ...
```

## Quickstart

### 1. Bootstrap the environment

```bash
bash setup.sh \
    --train-input /path/to/div2k \
    --test-input  /path/to/kodak \
    --output      runs \
    --gpus        0,1            # or "all" / "0" / "cpu"
source .walloc_env
```

`setup.sh` installs `compressai`, `geomloss`, `pytorch-msssim`, `tensorboard`,
then writes `.walloc_env` with:

| variable | purpose |
|----------|---------|
| `WALLOC_TRAIN_INPUT`   | default for `--data_root` |
| `WALLOC_TEST_INPUT`    | default for `--test_root` |
| `WALLOC_OUTPUT_DIR`    | default for `--out_dir` |
| `WALLOC_NUM_GPUS`      | default for `--num_gpus` (0 = auto) |
| `CUDA_VISIBLE_DEVICES` | from `--gpus` |
| `PYTHONPATH`           | repo root so `LTC` and `scripts` are importable |

CLI overrides always win over env defaults. Skip pip with `--no-install`.

### 2. Tests (~3 min)

```bash
pytest tests/test_walloc.py -v
```

Each test maps to a load-bearing piece of theory (crypto lemma, Prop 3.3
PD/SD distortion decomposition, QSD per-coordinate volume correction,
sliced-W_2 closed form on Gaussians, E8 coset geometry).

### 3. Train one config

```bash
python scripts/train.py \
    --dither_mode sd --lmbda 0.013 --lmbda_p 0.5 \
    --epochs 25 --batch_size 16 --patch_size 256
```

Outputs land in `$WALLOC_OUTPUT_DIR/<tag>/`:

```
sd_l0.013_lp0.5/
├── model.pt              # latest
├── model_best.pt         # best test loss
├── metrics.jsonl         # per-step log
├── summary.json
└── tb/                   # TensorBoard scalars + per-epoch input/recon/residual
```

### 4. Eval a checkpoint

```bash
python scripts/eval.py \
    --checkpoint runs/sd_l0.013_lp0.5/model_best.pt \
    --out_json   runs/sd_l0.013_lp0.5/eval_kodak.json \
    --tb_dir     runs/sd_l0.013_lp0.5/tb_eval
```

### 5. Run the full Phase-1 gate

```bash
python scripts/run_gate.py --epochs 20
```

7 trains + 7 evals, aggregates into `GATE_RESULTS.json` + a pass/kill memo
against the planning-doc criteria (P1: SD beats PD on perception; P2:
monotone in lambda_p; P3: det cannot reach SD's floor). Re-running picks
up where it left off (uses `summary.json` / `eval_kodak.json` as skip
markers).

### 6. TensorBoard

```bash
tensorboard --logdir $WALLOC_OUTPUT_DIR
```

Scalars: `train/{loss,bpp,mse,psnr,perc,aux,lr}`, `test/{loss,bpp,mse,psnr,perc,aux}`.
Images: `test/input`, `test/recon`, `test/residual_abs_norm`,
plus per-image `eval/<name>/input_vs_recon` and `eval/<name>/residual_abs_norm`.

## Kaggle

Open `notebooks/walloc_kaggle.ipynb` on Kaggle with **GPU T4 x 2** +
**Internet On**, attach a DIV2K (or CLIC) + Kodak dataset, then Run All.
The notebook calls `setup.sh` with `/kaggle/working/walloc_gate` as the
output dir, runs the gate matrix, then launches embedded TensorBoard.
