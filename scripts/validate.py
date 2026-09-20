"""Phase 3 - hold-out validation + confidence-threshold tuning.            Owner: Member 2

    python scripts/validate.py --config configs/config.yaml
"""
from _common import parse_args

from smartcheckout.data import dataset, yolo_format
from smartcheckout.detection import load_model
from smartcheckout.evaluation import validation
from smartcheckout.reporting import figures

if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__)
    ctx = dataset.load_context(cfg, paths)
    data_yaml = yolo_format.ensure_yolo_dataset(ctx, paths)
    model = load_model(paths.best)
    metrics, table = validation.validate_holdout(model, cfg, paths, data_yaml, ctx.meta_names)
    print("Hold-out:", {k: round(v, 4) for k, v in metrics.items()})
    print(table.to_string())
    pipe_cfg, curve = validation.tune_threshold(model, ctx.hold, cfg, paths)
    figures.threshold_curve(curve, pipe_cfg["conf"], paths.figures / "conf_threshold.png")
