"""Phase 6b - export the model for local CPU use (+ rpc17_export.zip).     Owner: Member 2

    python scripts/export_model.py --config configs/config.yaml
"""
from _common import parse_args

from smartcheckout.data import dataset
from smartcheckout.export import export_model

if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__)
    ctx = dataset.load_context(cfg, paths)
    export_model(cfg, paths, check_images=list(ctx.test.path.unique()[:3]))
