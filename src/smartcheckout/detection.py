"""
Modules B + C - product detection and 17-class classification in one YOLO pass,
plus the full inference pipeline (A -> B/C) used by evaluation and the local demo.

Owner: Member 1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .preprocessing import preprocess_image

DEFAULT_PIPELINE_CONFIG = dict(
    imgsz=1024,          # YOLO input size
    conf=0.40,           # confidence threshold (tuned on the hold-out by evaluation/validation.py)
    iou=0.70,            # NMS IoU threshold (kept high: products in hard scenes overlap)
    agnostic_nms=True,   # one box per product even if two categories compete for it
    preprocess=dict(max_side=2000, denoise=False, clahe=False),
)


def detect_and_classify(model, img: np.ndarray, meta_names, imgsz: int = 1024, conf: float = 0.4,
                        iou: float = 0.7, agnostic_nms: bool = True, device=None) -> list[dict]:
    """One detection per product: box (x1, y1, x2, y2), meta (class index), meta_name, meta_conf."""
    r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, agnostic_nms=agnostic_nms,
                      device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    scores = r.boxes.conf.cpu().numpy()
    return [dict(box=[float(v) for v in b], meta=int(c), meta_name=meta_names[int(c)], meta_conf=float(s))
            for b, c, s in zip(boxes, cls, scores)]


class Pipeline17:
    """image -> Module A (preprocess) -> Modules B+C (YOLO) -> list of detections."""

    def __init__(self, model, meta_names, config: dict | None = None, device="cpu"):
        self.model = model
        self.meta_names = list(meta_names)
        self.cfg = json.loads(json.dumps(DEFAULT_PIPELINE_CONFIG))
        if config:
            pp = dict(self.cfg["preprocess"])
            pp.update(config.get("preprocess", {}))
            self.cfg.update(config)
            self.cfg["preprocess"] = pp
        self.device = "cpu" if str(device) == "cpu" else 0

    def __call__(self, img_bgr: np.ndarray, preprocess: dict | None = None):
        pp = dict(self.cfg["preprocess"])
        pp.update(preprocess or {})
        img = preprocess_image(img_bgr, **pp)
        dets = detect_and_classify(self.model, img, self.meta_names, self.cfg["imgsz"], self.cfg["conf"],
                                   self.cfg["iou"], self.cfg["agnostic_nms"], self.device)
        return img, dets


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))


def load_pipeline(weights_dir, device="cpu") -> Pipeline17:
    """Load model.pt, categories.json and pipeline_config.json from an export folder."""
    wd = Path(weights_dir)
    meta_names = json.loads((wd / "categories.json").read_text(encoding="utf-8"))["meta_names"]
    cfg_path = wd / "pipeline_config.json"
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return Pipeline17(load_model(wd / "model.pt"), meta_names, config, device)
