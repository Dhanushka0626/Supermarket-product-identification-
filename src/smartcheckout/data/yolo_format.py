"""
YOLO-format dataset (images as symlinks + one label file per image, 17 classes).

Owner: Member 1

YOLO label line:  <class> <x_center> <y_center> <width> <height>   (all relative to image size)
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


def write_split(yolo_dir: Path, part: str, df: pd.DataFrame) -> None:
    (yolo_dir / "images" / part).mkdir(parents=True, exist_ok=True)
    (yolo_dir / "labels" / part).mkdir(parents=True, exist_ok=True)
    for _, g in tqdm(df.groupby("image_id"), desc=f"YOLO {part}"):
        src = Path(g.path.iloc[0])
        dst = yolo_dir / "images" / part / src.name
        if not dst.exists():
            os.symlink(src, dst)
        W, H = float(g.width.iloc[0]), float(g.height.iloc[0])
        lines = []
        for m, x, y, w, h in g[["meta", "x", "y", "w", "h"]].to_numpy():
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"{int(m)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (yolo_dir / "labels" / part / (src.stem + ".txt")).write_text("\n".join(lines))


def names_block(meta_names) -> str:
    return "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(meta_names))


def ensure_yolo_dataset(ctx, paths) -> Path:
    """Build the YOLO folders once per session (they live in tmp and are rebuilt when missing)."""
    if not paths.data_yaml.exists():
        write_split(paths.yolo, "train", ctx.train)
        write_split(paths.yolo, "val", ctx.hold)
        write_split(paths.yolo, "test", ctx.test)
        paths.data_yaml.write_text(f"path: {paths.yolo}\ntrain: images/train\nval: images/val\n"
                                   f"test: images/test\n" + names_block(ctx.meta_names))
    return paths.data_yaml


def yolo_image_path(paths, image_path: str, in_holdout: bool) -> str:
    """Where a val2019 image lives inside the YOLO folder (train part or hold-out part)."""
    return str(paths.yolo / "images" / ("val" if in_holdout else "train") / Path(image_path).name)


def write_fold_dataset(fold_dir: Path, train_paths, val_paths, meta_names) -> Path:
    """A fold is described by two text files listing image paths (labels are found automatically)."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "train.txt").write_text("\n".join(train_paths))
    (fold_dir / "val.txt").write_text("\n".join(val_paths))
    yaml_path = fold_dir / "data.yaml"
    yaml_path.write_text(f"path: {fold_dir}\ntrain: {fold_dir / 'train.txt'}\nval: {fold_dir / 'val.txt'}\n"
                         + names_block(meta_names))
    return yaml_path
