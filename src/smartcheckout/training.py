"""
Model training (final model and cross-validation fold models).

Owner: Member 1

The model is a COCO-pretrained YOLO (default YOLO11-m) fine-tuned on the 17 classes.
Training saves a checkpoint every epoch; calling a train function again resumes an
interrupted run, and a finished run is skipped (DONE marker).
"""
from __future__ import annotations

from pathlib import Path

# Augmentation: products are photographed top-down, so both flips are realistic; colour jitter
# covers lighting changes; mosaic mixes several checkout scenes. No rotation (boxes would loosen).
AUGMENTATION = dict(fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=1.0, hsv_h=0.01, hsv_s=0.5, hsv_v=0.3)


def _train_or_resume(run_dir: Path, start_new) -> None:
    last, done = run_dir / "weights" / "last.pt", run_dir / "DONE"
    if done.exists():
        print(f"{run_dir.name}: already trained - skipping.")
        return
    from ultralytics import YOLO
    if last.exists():
        print(f"{run_dir.name}: resuming from {last}")
        try:
            YOLO(str(last)).train(resume=True)
        except Exception as e:           # the run had already finished
            print("Resume not possible:", e)
    else:
        start_new()
    done.touch()


def train_final(cfg: dict, paths, data_yaml: Path) -> Path:
    """Phase 2: train the final model on the val2019 training part (hold-out used for monitoring)."""
    from ultralytics import YOLO
    run_dir = paths.runs / "rpc17"

    def start():
        YOLO(cfg["model"]).train(
            data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"],
            patience=cfg["patience"], device=cfg["device"], workers=cfg["workers"],
            project=str(paths.runs), name="rpc17", exist_ok=True, seed=cfg["seed"],
            cos_lr=True, optimizer="AdamW", lr0=0.001, close_mosaic=10, plots=True, **AUGMENTATION)

    _train_or_resume(run_dir, start)
    return paths.best


def train_fold(cfg: dict, fold_dir: Path, data_yaml: Path, k: int) -> Path:
    """Cross-validation: train one fold model. No early stopping, and the LAST checkpoint is used,
    so the fold's validation images never influence training or model selection."""
    from ultralytics import YOLO
    run_dir = fold_dir / "train"

    def start():
        YOLO(cfg["model"]).train(
            data=str(data_yaml), imgsz=cfg["imgsz"], epochs=cfg["cv_epochs"], batch=cfg["batch"],
            patience=cfg["cv_epochs"] + 1, device=cfg["device"], workers=cfg["workers"],
            project=str(fold_dir), name="train", exist_ok=True, seed=cfg["seed"] + k,
            cos_lr=True, optimizer="AdamW", lr0=0.001, close_mosaic=min(10, cfg["cv_epochs"] // 2),
            plots=False, **AUGMENTATION)

    _train_or_resume(run_dir, start)
    return run_dir / "weights" / "last.pt"
