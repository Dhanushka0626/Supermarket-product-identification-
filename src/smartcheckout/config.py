"""
Project configuration and shared paths.

Owner: Member 1 (shared by both members - change only through a reviewed pull request)

All settings live in configs/config.yaml. Any setting can be overridden on the command line:
    python scripts/train.py --config configs/config.yaml --set epochs=10 batch=4
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

DEFAULTS = dict(
    # locations
    input_root="/kaggle/input",      # the RPC dataset is searched for below this folder
    work_dir="/kaggle/working",      # everything that must be kept (weights, results, figures)
    tmp_dir="/tmp/rpc17",            # rebuildable data (YOLO symlinks + label files)
    resume_from=None,                # output folder of a previous run to continue from
    seed=42,

    # data split
    holdout_frac=0.10,               # share of val2019 images held out for validation / tuning
    test_limit=None,                 # None = all 24,000 test2019 images

    # model (Modules B + C)
    model="yolo11m.pt",
    imgsz=1024,
    epochs=60,
    batch=8,
    patience=20,
    device=0,                        # 0 = first GPU, [0, 1] = two GPUs, "cpu"
    iou=0.70,
    workers=4,

    # cross-validation
    run_cv=True,
    cv_folds=5,
    cv_epochs=20,
    cv_folds_to_run=None,            # None = all folds, e.g. [0, 1, 2]
)


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Defaults <- YAML file <- command-line overrides ("key=value")."""
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting '{key}'. Valid settings: {', '.join(DEFAULTS)}")
        cfg[key] = _parse_value(value)
    return cfg


def get_paths(cfg: dict) -> SimpleNamespace:
    """All folders and files shared between the pipeline steps."""
    work, tmp = Path(cfg["work_dir"]), Path(cfg["tmp_dir"])
    p = SimpleNamespace(
        work=work, tmp=tmp,
        data=work / "data",              # annotation tables, splits, folds
        export=work / "export",          # files needed by the local demo
        results=work / "results",        # tables and metrics
        figures=work / "figures",
        cv=work / "cv",
        runs=work / "runs",
        outcomes=work / "outcomes",
        yolo=tmp / "yolo",
    )
    p.best = p.runs / "rpc17" / "weights" / "best.pt"
    p.last = p.runs / "rpc17" / "weights" / "last.pt"
    p.data_yaml = p.yolo / "rpc17.yaml"
    p.categories = p.export / "categories.json"
    p.pipeline_config = p.export / "pipeline_config.json"
    for d in (p.work, p.tmp, p.data, p.export, p.results, p.figures, p.cv, p.runs):
        d.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def restore_previous(cfg: dict, paths: SimpleNamespace) -> None:
    """Copy the outputs of an earlier (Kaggle) session so finished steps are skipped."""
    if not cfg.get("resume_from"):
        return
    src = Path(cfg["resume_from"])
    if not src.exists():
        raise FileNotFoundError(f"resume_from folder not found: {src}")
    marker = paths.work / ".restored_from"
    if marker.exists() and marker.read_text() == str(src.resolve()):
        return                         # already restored in this session - never overwrite newer results
    for item in ("data", "runs", "cv", "export", "results", "figures"):
        if (src / item).is_dir():
            shutil.copytree(src / item, paths.work / item, dirs_exist_ok=True)
    marker.write_text(str(src.resolve()))
    print(f"Restored previous outputs from {src}")


def setup(config_path=None, overrides=None):
    """Convenience used by every script: config + paths + seed + optional restore."""
    cfg = load_config(config_path, overrides)
    paths = get_paths(cfg)
    seed_everything(cfg["seed"])
    restore_previous(cfg, paths)
    return cfg, paths
