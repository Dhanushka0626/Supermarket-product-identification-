"""Phase 5 - K-fold cross-validation on val2019.                           Owner: Member 2

Each fold trains a new model; finished folds are skipped, so folds can be split over sessions:
    python scripts/cross_validate.py --set cv_folds_to_run=[0,1,2]
    python scripts/cross_validate.py --set cv_folds_to_run=[3,4]
Only summarise folds that are already finished:
    python scripts/cross_validate.py --summary-only
"""
import json

from _common import device_of, parse_args

from smartcheckout.data import dataset, yolo_format
from smartcheckout.evaluation import cross_validation as cv
from smartcheckout.evaluation import validation
from smartcheckout.reporting import figures


def extra(ap):
    ap.add_argument("--summary-only", action="store_true", help="do not train, only summarise finished folds")


def final_test_numbers(paths) -> dict:
    out = {}
    if (paths.results / "test_map.json").exists():
        out.update(json.loads((paths.results / "test_map.json").read_text()))
    if (paths.results / "test_end_to_end.csv").exists():
        import pandas as pd
        out.update(pd.read_csv(paths.results / "test_end_to_end.csv", index_col=0).loc["all"].to_dict())
    return out


if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__, extra)
    ctx = dataset.load_context(cfg, paths)
    if cfg["run_cv"] and not args.summary_only:
        yolo_format.ensure_yolo_dataset(ctx, paths)
        cv.run_cross_validation(ctx, cfg, paths, validation.load_pipeline_config(paths), device_of(cfg))
    results = cv.collect_results(paths)
    print(f"Finished folds: {[r['fold'] for r in results]} of {cfg['cv_folds']}")
    summary = cv.summarize(results, ctx.meta_names, final_test_numbers(paths), paths)
    if summary is not None:
        print(summary["summary"].round(4).to_string())
        figures.cv_summary(summary, paths.figures / "cv_summary.png")
        figures.cv_per_class(summary["per_class"], paths.figures / "cv_per_class_ap50.png")
