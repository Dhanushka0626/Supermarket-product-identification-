"""
Final evaluation on test2019 (Phase 4). test2019 is never used for training or tuning.

Owner: Member 2

Results are cached, so a later session (or scripts/build_outcomes.py) can reuse them without
re-running the model on 24,000 images.
"""
from __future__ import annotations

import json

import pandas as pd

from ..detection import Pipeline17
from .metrics import by_level, evaluate_system, make_groups, per_class_prf
from .validation import per_class_ap_table, summary_metrics


def detection_metrics(model, cfg: dict, paths, data_yaml, meta_names, force: bool = False):
    cache = paths.results / "test_map.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text()), pd.read_csv(paths.results / "test_ap_per_class.csv", index_col=0)
    m = model.val(data=str(data_yaml), split="test", imgsz=cfg["imgsz"], batch=cfg["batch"], device=cfg["device"],
                  iou=cfg["iou"], plots=True, project=str(paths.runs), name="test", exist_ok=True, verbose=False)
    metrics, table = summary_metrics(m), per_class_ap_table(m, meta_names)
    table.to_csv(paths.results / "test_ap_per_class.csv")
    cache.write_text(json.dumps(metrics, indent=2))
    return metrics, table


def system_metrics(model, ctx, cfg: dict, paths, pipe_cfg: dict, device, force: bool = False) -> dict:
    cache = paths.results / "test_eval.pkl"
    if cache.exists() and not force:
        ev = pd.read_pickle(cache)
    else:
        pipe = Pipeline17(model, ctx.meta_names, pipe_cfg, device=device)
        ev = evaluate_system(pipe, make_groups(ctx.test), ctx.n_classes, cfg["workers"], "full system on test2019")
        pd.to_pickle(ev, cache)
    ev["rec"].to_csv(paths.results / "test_per_image.csv", index=False)
    e2e = by_level(ev["rec"])
    e2e.to_csv(paths.results / "test_end_to_end.csv")
    per_class = per_class_prf(ev["per_cls"], ctx.meta_names)
    per_class.to_csv(paths.results / "test_per_class.csv", index=False)
    (paths.results / "test_timing.json").write_text(json.dumps({"ms_per_image": ev["ms_per_image"], "device": str(device)}))
    return dict(ev=ev, e2e=e2e, per_class=per_class)
