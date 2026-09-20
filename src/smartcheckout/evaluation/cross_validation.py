"""
K-fold cross-validation (Phase 5).

Owner: Member 2

1. All val2019 images are divided into K folds (by image, stratified by difficulty level).
2. For every fold k a new model is trained on the other K-1 folds and evaluated on fold k.
3. Every fold is scored with exactly the same metrics as the test set.
4. Results: mean, standard deviation and 95 % confidence interval (t-distribution).

Each fold saves fold_results.json, so folds can be split over several Kaggle sessions.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import splits, yolo_format
from ..detection import Pipeline17, load_model
from ..training import train_fold
from .metrics import aggregate, by_level, evaluate_system, make_groups, per_class_prf
from .validation import summary_metrics

CV_METRICS = ["precision", "recall", "mAP50", "mAP50_95", "classification_acc", "product_level_acc",
              "det_precision", "det_recall", "count_acc", "cAcc", "ACD"]
SYSTEM_METRICS = ["classification_acc", "product_level_acc", "det_precision", "det_recall", "count_acc", "cAcc", "ACD"]


def fold_table(ctx, paths, cfg) -> pd.DataFrame:
    """One row per val2019 image with its fold and its path inside the YOLO folder."""
    fold_of = splits.load_or_make_folds(paths, ctx.ann["val"], cfg["cv_folds"], cfg["seed"])
    tab = ctx.ann["val"].groupby("image_id").agg(path=("path", "first"), level=("level", "first"))
    tab["fold"] = tab.index.map(fold_of)
    tab["yolo_path"] = [yolo_format.yolo_image_path(paths, p, i in ctx.holdout_ids) for i, p in zip(tab.index, tab.path)]
    return tab


def fold_sizes(tab: pd.DataFrame, ann_val: pd.DataFrame) -> pd.DataFrame:
    sizes = pd.crosstab(tab.fold, tab.level)
    sizes["images"] = sizes.sum(axis=1)
    sizes["products"] = ann_val.assign(fold=ann_val.image_id.map(tab.fold)).groupby("fold").size()
    sizes.loc["total"] = sizes.sum()
    return sizes


def run_fold(k: int, tab, ctx, cfg: dict, paths, pipe_cfg: dict, device) -> dict:
    fdir = paths.cv / f"fold{k}"
    res_file = fdir / "fold_results.json"
    if res_file.exists():
        print(f"fold {k}: already finished - loading results")
        return json.loads(res_file.read_text())

    tr, va = tab[tab.fold != k], tab[tab.fold == k]
    data_yaml = yolo_format.write_fold_dataset(fdir, tr.yolo_path, va.yolo_path, ctx.meta_names)
    weights = train_fold(cfg, fdir, data_yaml, k)

    model = load_model(weights)
    vm = model.val(data=str(data_yaml), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"], device=cfg["device"],
                   iou=cfg["iou"], plots=False, project=str(fdir), name="val", exist_ok=True, verbose=False)
    ap50 = {ctx.meta_names[c]: float(vm.box.ap50[i]) for i, c in enumerate(vm.box.ap_class_index)}
    groups = make_groups(ctx.ann["val"][ctx.ann["val"].image_id.isin(set(va.index))])
    ev = evaluate_system(Pipeline17(model, ctx.meta_names, pipe_cfg, device=device), groups, ctx.n_classes,
                         cfg["workers"], f"fold {k} evaluation")
    agg = aggregate(ev["rec"])
    out = dict(fold=k, train_images=len(tr), val_images=len(va), **summary_metrics(vm),
               **{m: float(agg[m]) for m in SYSTEM_METRICS},
               by_level=by_level(ev["rec"]).reset_index().to_dict(orient="records"),
               ap50_per_class=ap50,
               f1_per_class=per_class_prf(ev["per_cls"], ctx.meta_names).set_index("category")["f1"].to_dict())
    res_file.write_text(json.dumps(out, indent=2, default=float))
    return out


def run_cross_validation(ctx, cfg: dict, paths, pipe_cfg: dict, device) -> list:
    tab = fold_table(ctx, paths, cfg)
    fold_sizes(tab, ctx.ann["val"]).to_csv(paths.cv / "fold_sizes.csv")
    todo = cfg["cv_folds_to_run"] if cfg["cv_folds_to_run"] is not None else range(cfg["cv_folds"])
    for k in todo:
        run_fold(int(k), tab, ctx, cfg, paths, pipe_cfg, device)
    return collect_results(paths)


def collect_results(paths) -> list:
    """All finished folds, including those finished in earlier sessions."""
    results = [json.loads(f.read_text()) for f in paths.cv.glob("fold*/fold_results.json")]
    return sorted(results, key=lambda r: r["fold"])


def summarize(results: list, meta_names, final_test: dict | None = None, paths=None) -> dict | None:
    """Mean, std, min, max and 95 % CI over folds (+ per class and per difficulty level)."""
    if not results:
        return None
    from scipy import stats

    folds = pd.DataFrame([{k: r[k] for k in ["fold", "train_images", "val_images"] + CV_METRICS} for r in results])
    n = len(folds)

    def row(x):
        m, sd = x.mean(), (x.std(ddof=1) if n > 1 else np.nan)
        h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        return pd.Series(dict(mean=m, std=sd, min=x.min(), max=x.max(), ci95_low=m - h, ci95_high=m + h))

    summary = folds[CV_METRICS].apply(row).T
    ratio = [m for m in summary.index if m != "ACD"]                  # rates cannot leave [0, 1]
    summary.loc[ratio, ["ci95_low", "ci95_high"]] = summary.loc[ratio, ["ci95_low", "ci95_high"]].clip(0, 1)
    summary["final_model_test2019"] = [(final_test or {}).get(m, np.nan) for m in summary.index]
    summary.index.name = "metric"

    ap = pd.DataFrame([r["ap50_per_class"] for r in results]).reindex(columns=meta_names)
    f1 = pd.DataFrame([r["f1_per_class"] for r in results]).reindex(columns=meta_names)
    per_class = pd.DataFrame({"AP50_mean": ap.mean(), "AP50_std": ap.std(ddof=1),
                              "F1_mean": f1.mean(), "F1_std": f1.std(ddof=1)}).round(4)
    per_class.index.name = "category"

    lvl = pd.DataFrame([dict(fold=r["fold"], **row_) for r in results for row_ in r["by_level"]])
    by_lvl = lvl.groupby("level")[["classification_acc", "product_level_acc", "count_acc", "cAcc", "ACD"]].agg(["mean", "std"]).round(4)
    by_lvl.columns = [f"{a}_{b}" for a, b in by_lvl.columns]

    out = dict(folds=folds, summary=summary, per_class=per_class, by_level=by_lvl)
    if paths is not None:
        folds.to_csv(paths.cv / "cv_folds.csv", index=False)
        summary.to_csv(paths.cv / "cv_summary.csv")
        per_class.to_csv(paths.cv / "cv_per_class.csv")
        by_lvl.to_csv(paths.cv / "cv_by_level.csv")
    ca = summary.loc["classification_acc"]
    print(f"Classification accuracy over {n} folds: {ca['mean']:.4f} ± {ca['std']:.4f} "
          f"(95% CI {ca['ci95_low']:.4f} - {ca['ci95_high']:.4f}) -> 80% target "
          f"{'MET' if ca['mean'] >= 0.8 else 'NOT met'}")
    return out
