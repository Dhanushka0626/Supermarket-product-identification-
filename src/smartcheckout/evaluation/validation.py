"""
Hold-out validation and confidence-threshold tuning (Phase 3).

Owner: Member 2

The threshold is chosen to maximise checkout accuracy (cAcc) on the hold-out images: a checkout
system is only useful when the whole shopping list (count per category) is right.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..detection import DEFAULT_PIPELINE_CONFIG


def per_class_ap_table(val_metrics, meta_names) -> pd.DataFrame:
    """AP50 and AP50-95 per class from an Ultralytics validation result."""
    rows = {meta_names[c]: dict(AP50=val_metrics.box.ap50[i], AP50_95=val_metrics.box.ap[i])
            for i, c in enumerate(val_metrics.box.ap_class_index)}
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(meta_names).round(4)
    df.index.name = "category"
    return df


def summary_metrics(val_metrics) -> dict:
    return dict(precision=float(val_metrics.box.mp), recall=float(val_metrics.box.mr),
                mAP50=float(val_metrics.box.map50), mAP50_95=float(val_metrics.box.map))


def validate_holdout(model, cfg: dict, paths, data_yaml, meta_names):
    """mAP on the hold-out + per-class AP + Ultralytics confusion matrix image."""
    m = model.val(data=str(data_yaml), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"],
                  device=cfg["device"], iou=cfg["iou"], plots=True, project=str(paths.runs),
                  name="val_holdout", exist_ok=True, verbose=False)
    metrics, table = summary_metrics(m), per_class_ap_table(m, meta_names)
    (paths.results / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2))
    table.to_csv(paths.results / "holdout_ap_per_class.csv")
    return metrics, table


def tune_threshold(model, hold: pd.DataFrame, cfg: dict, paths):
    """Sweep the confidence threshold on the hold-out images and save pipeline_config.json."""
    ho = hold.groupby("image_id").agg(path=("path", "first"), metas=("meta", list)).reset_index()
    preds = []
    for i in tqdm(range(0, len(ho), 16), desc="hold-out prediction"):
        for r in model.predict(ho.path.iloc[i:i + 16].tolist(), imgsz=cfg["imgsz"], conf=0.05, iou=cfg["iou"],
                               agnostic_nms=True, device=cfg["device"], verbose=False):
            preds.append((r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))

    thresholds = np.round(np.arange(0.10, 0.91, 0.025), 3)
    rows = []
    for t in thresholds:
        rows.append(dict(
            threshold=t,
            count_acc=np.mean([(c >= t).sum() == len(g) for (c, _), g in zip(preds, ho.metas)]),
            cAcc=np.mean([Counter(k[c >= t].tolist()) == Counter(g) for (c, k), g in zip(preds, ho.metas)])))
    curve = pd.DataFrame(rows)
    best = curve.loc[curve.cAcc.idxmax()]
    curve.to_csv(paths.results / "threshold_tuning.csv", index=False)

    pipe_cfg = json.loads(json.dumps(DEFAULT_PIPELINE_CONFIG))
    pipe_cfg.update(imgsz=cfg["imgsz"], conf=float(best.threshold), iou=cfg["iou"])
    paths.pipeline_config.write_text(json.dumps(pipe_cfg, indent=2))
    print(f"Best confidence threshold = {best.threshold}  (hold-out cAcc {best.cAcc:.4f}, "
          f"count accuracy {best.count_acc:.4f})")
    return pipe_cfg, curve


def load_pipeline_config(paths) -> dict:
    if not paths.pipeline_config.exists():
        raise FileNotFoundError("pipeline_config.json missing - run scripts/validate.py first")
    return json.loads(paths.pipeline_config.read_text())
