"""Phase 4 - final evaluation on test2019 (never used for training or tuning).   Owner: Member 2

    python scripts/evaluate_test.py --config configs/config.yaml [--force]
"""
from _common import device_of, parse_args

from smartcheckout.data import dataset, yolo_format
from smartcheckout.detection import Pipeline17, load_model
from smartcheckout.evaluation import test_eval, validation
from smartcheckout.evaluation.metrics import make_groups
from smartcheckout.reporting import figures


def extra(ap):
    ap.add_argument("--force", action="store_true", help="recompute even if cached results exist")


if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__, extra)
    ctx = dataset.load_context(cfg, paths)
    data_yaml = yolo_format.ensure_yolo_dataset(ctx, paths)
    model, device = load_model(paths.best), device_of(cfg)
    pipe_cfg = validation.load_pipeline_config(paths)

    det, ap_table = test_eval.detection_metrics(model, cfg, paths, data_yaml, ctx.meta_names, args.force)
    print("test2019 detection:", {k: round(v, 4) for k, v in det.items()})
    print(ap_table.to_string())

    res = test_eval.system_metrics(model, ctx, cfg, paths, pipe_cfg, device, args.force)
    print(res["e2e"].round(4).to_string())
    print(res["per_class"].to_string())

    figures.confusion(res["ev"]["pair_true"], res["ev"]["pair_pred"], ctx.meta_names,
                      "Category confusion matrix - test2019 (located products, row-normalised)",
                      paths.figures / "cm_test.png")
    figures.examples(Pipeline17(model, ctx.meta_names, pipe_cfg, device), make_groups(ctx.test),
                     ctx.meta_names, paths.figures)
