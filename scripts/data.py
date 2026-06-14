"""
Dataset helpers shared by training and eval scripts.

- FlatImageFolder: recursively reads images from any folder. No train/test
  split conventions required (CompressAI's ImageFolder needs train/ + test/
  subdirs, which Kaggle / arbitrary dumps rarely follow).
- autodetect_dataset: picks a sensible image dir when none is provided —
  honours $WALLOC_TRAIN_INPUT / $WALLOC_TEST_INPUT first, then scans
  /kaggle/input by common dataset names.
- short_dataset_summary: quick diagnostics for printing at startup.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp", ".BMP")


class FlatImageFolder(Dataset):
    """Recursively read every image under `root`.

    `min_size` filters out tiny thumbnails some datasets ship alongside HR
    images (e.g. DIV2K LR sets).
    """

    def __init__(
        self,
        root: str,
        transform=None,
        min_size: int = 0,
        max_images: Optional[int] = None,
    ):
        self.root = root
        self.transform = transform
        paths: List[str] = []
        for ext in IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        paths = sorted(set(paths))
        if min_size > 0:
            keep = []
            for p in paths:
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                    if min(w, h) >= min_size:
                        keep.append(p)
                except Exception:
                    pass
            paths = keep
        if max_images is not None:
            paths = paths[:max_images]
        if not paths:
            raise FileNotFoundError(f"no images under {root}")
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


def autodetect_dataset(
    user_path: Optional[str] = None,
    purpose: str = "train",
) -> str:
    """Find a usable dataset directory.

    Search order:
      1. `user_path` if it exists.
      2. $WALLOC_TRAIN_INPUT / $WALLOC_TEST_INPUT (set by setup.sh).
      3. /kaggle/input/<heuristic name match for purpose>.
      4. Fallback: first /kaggle/input/* that contains any image.

    `purpose` in {'train', 'test'}.
    """
    if user_path and os.path.isdir(user_path):
        return user_path

    env_key = "WALLOC_TRAIN_INPUT" if purpose == "train" else "WALLOC_TEST_INPUT"
    env_path = os.environ.get(env_key)
    if env_path and os.path.isdir(env_path):
        return env_path

    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        raise FileNotFoundError(
            f"no dataset found. Set ${env_key} (e.g. via setup.sh) or pass "
            f"--{purpose[:4]}_root. user_path={user_path!r}"
        )

    candidates = [p for p in kaggle_input.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("/kaggle/input is empty — attach a Kaggle dataset first")

    if purpose == "test":
        patterns = ("kodak", "clic2020", "clic")
    else:
        patterns = ("div2k", "div", "clic2020", "clic", "imagenet", "openimages")

    for pat in patterns:
        for c in candidates:
            if pat in c.name.lower():
                return str(c)

    for c in candidates:
        for ext in IMAGE_EXTS:
            if next(c.rglob(f"*{ext}"), None) is not None:
                print(
                    f"[autodetect_dataset] no name match for {purpose!r}; "
                    f"falling back to {c}"
                )
                return str(c)

    raise FileNotFoundError(
        f"could not autodetect {purpose} dataset. Candidates: "
        f"{[c.name for c in candidates]}. Set ${env_key} or pass --{purpose[:4]}_root."
    )


def short_dataset_summary(root: str) -> Tuple[int, Tuple[int, int]]:
    """Return (n_images, (median_w, median_h)) for startup diagnostics."""
    paths: List[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    sample = paths[:50]
    if not sample:
        return 0, (0, 0)
    ws, hs = [], []
    for p in sample:
        try:
            with Image.open(p) as im:
                ws.append(im.width); hs.append(im.height)
        except Exception:
            pass
    ws.sort(); hs.sort()
    mw = ws[len(ws) // 2] if ws else 0
    mh = hs[len(hs) // 2] if hs else 0
    return len(paths), (mw, mh)
