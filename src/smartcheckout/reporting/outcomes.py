"""
Outcomes package: every result of the project in one folder (and zip) for the report and viva.

Owner: Member 2

Reads only files saved by the earlier steps, so it can be rebuilt at any time:
    outcomes/
    ├── outcomes_report.md     readable summary of all results
    ├── outcomes_summary.json  every number in machine-readable form
    ├── outcomes.xlsx          one sheet per table
    ├── tables/                the same tables as CSV
    └── figures/               all plots
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import pandas as pd


def _read_csv(path, **kw):
    return pd.read_csv(path, **kw) if path.exists() else None


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    d = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    fmt = lambda v: f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def build(cfg: dict, paths) -> dict:
    r, cvd = paths.results, paths.cv
    tables = {
        "class_definition": _read_csv(r / "class_definition.csv", index_col=0),
        "class_counts": _read_csv(r / "class_counts.csv", index_col=0),
        "holdout_ap_per_class": _read_csv(r / "holdout_ap_per_class.csv", index_col=0),
        "threshold_tuning": _read_csv(r / "threshold_tuning.csv"),
        "test_end_to_end": _read_csv(r / "test_end_to_end.csv", index_col=0),
        "test_per_class": _read_csv(r / "test_per_class.csv", index_col=0),
        "test_ap_per_class": _read_csv(r / "test_ap_per_class.csv", index_col=0),
        "cv_fold_sizes": _read_csv(cvd / "fold_sizes.csv", index_col=0),
        "cv_folds": _read_csv(cvd / "cv_folds.csv", index_col=0),
        "cv_summary": _read_csv(cvd / "cv_summary.csv", index_col=0),
        "cv_per_class": _read_csv(cvd / "cv_per_class.csv", index_col=0),
        "cv_by_level": _read_csv(cvd / "cv_by_level.csv", index_col=0),
    }
    tables = {k: v for k, v in tables.items() if v is not None}
    hold = _read_json(r / "holdout_metrics.json")
    test_map = _read_json(r / "test_map.json")
    timing = _read_json(r / "test_timing.json")
    pipe_cfg = _read_json(paths.pipeline_config)
    categories = _read_json(paths.categories) or {}

    out = paths.outcomes
    if out.exists():
        shutil.rmtree(out)
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir()

    for name, df in tables.items():
        df.to_csv(out / "tables" / f"{name}.csv")
    try:
        with pd.ExcelWriter(out / "outcomes.xlsx") as xw:
            for name, df in tables.items():
                df.to_excel(xw, sheet_name=name[:31])
    except Exception as e:                       # openpyxl missing -> CSVs are still there
        print("Excel file skipped:", e)

    extra = [(paths.runs / "rpc17" / "results.png", "training_curves.png"),
             (paths.runs / "val_holdout" / "confusion_matrix_normalized.png", "holdout_confusion_matrix_yolo.png"),
             (paths.runs / "test" / "confusion_matrix_normalized.png", "test_confusion_matrix_yolo.png")]
    for f in paths.figures.glob("*"):
        shutil.copy(f, out / "figures" / f.name)
    for src, name in extra:
        if src.exists():
            shutil.copy(src, out / "figures" / name)

    e2e = tables.get("test_end_to_end")
    cvs = tables.get("cv_summary")
    test_acc = float(e2e.loc["all", "classification_acc"]) if e2e is not None else None
    summary = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"),
        classes=categories.get("meta_names"),
        model=dict(architecture=cfg["model"], imgsz=cfg["imgsz"], epochs=cfg["epochs"],
                   conf_threshold=(pipe_cfg or {}).get("conf"), iou=cfg["iou"]),
        holdout=hold, test2019_detection=test_map, timing=timing,
        test2019_end_to_end=None if e2e is None else e2e.round(4).reset_index().to_dict(orient="records"),
        cross_validation=None if cvs is None else dict(
            folds=len(tables["cv_folds"]), epochs_per_fold=cfg["cv_epochs"],
            summary=cvs.round(4).reset_index().to_dict(orient="records")),
        target_80pct=dict(test_classification_acc=test_acc,
                          met_on_test=None if test_acc is None else test_acc >= 0.8,
                          cv_mean=None if cvs is None else float(cvs.loc["classification_acc", "mean"])))
    (out / "outcomes_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    rep = [f"# Outcomes - 17-category YOLO model ({cfg['model']})", "", f"Generated: {summary['generated']}", ""]
    if "class_definition" in tables:
        rep += ["## Classes (from the dataset's `supercategory` field)", md_table(tables["class_definition"]), ""]
    if hold:
        rep += ["## Hold-out validation (val2019, 10 %)", md_table(pd.DataFrame([hold])), "",
                f"Tuned confidence threshold: **{(pipe_cfg or {}).get('conf')}**", ""]
    if test_map:
        rep += ["## Final test (test2019)", md_table(pd.DataFrame([test_map])), ""]
    if e2e is not None:
        rep += [md_table(e2e), ""]
        if timing:
            rep += [f"Average time: {timing['ms_per_image']:.0f} ms per image ({timing['device']})", ""]
    if "test_per_class" in tables:
        per_class = tables["test_per_class"]
        if "test_ap_per_class" in tables:
            per_class = per_class.join(tables["test_ap_per_class"])
        rep += ["### Per class (test2019)", md_table(per_class), ""]
    if cvs is not None:
        ca = cvs.loc["classification_acc"]
        rep += [f"## {len(tables['cv_folds'])}-fold cross-validation (val2019, {cfg['cv_epochs']} epochs per fold)",
                f"Classification accuracy: **{ca['mean']:.4f} ± {ca['std']:.4f}** "
                f"(95 % CI {ca['ci95_low']:.4f} - {ca['ci95_high']:.4f})", "",
                md_table(cvs), "", "### Per fold", md_table(tables["cv_folds"]), ""]
        for key, title in (("cv_per_class", "Per class (mean ± std over folds)"), ("cv_by_level", "Per difficulty level")):
            if key in tables:
                rep += [f"### {title}", md_table(tables[key]), ""]
    rep += ["## 80 % accuracy target"]
    if test_acc is not None:
        rep += [f"* test2019 classification accuracy: **{test_acc:.4f}** -> {'met' if test_acc >= 0.8 else 'not met'}"]
    if cvs is not None:
        ca = cvs.loc["classification_acc"]
        rep += [f"* cross-validation classification accuracy: **{ca['mean']:.4f} ± {ca['std']:.4f}** "
                f"(95 % CI lower bound {ca['ci95_low']:.4f}) -> {'met' if ca['mean'] >= 0.8 else 'not met'}"]
    (out / "outcomes_report.md").write_text("\n".join(rep), encoding="utf-8")

    zip_path = shutil.make_archive(str(paths.work / "rpc17_outcomes"), "zip", out)
    print("Outcomes ready:", zip_path)
    return summary
