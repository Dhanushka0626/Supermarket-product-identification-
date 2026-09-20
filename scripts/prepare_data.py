"""Phase 1 - data preparation.                                             Owner: Member 1

Reads the RPC dataset, defines the 17 classes from the dataset's `supercategory` field, saves the
annotation tables, the hold-out split, the cross-validation folds and the YOLO-format dataset.

    python scripts/prepare_data.py --config configs/config.yaml
"""
from _common import parse_args

from smartcheckout.data import dataset, yolo_format

if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__)
    ctx = dataset.prepare(cfg, paths)
    print("YOLO dataset:", yolo_format.ensure_yolo_dataset(ctx, paths))
    for name, df in (("train (val2019)", ctx.train), ("hold-out (val2019)", ctx.hold), ("test (test2019)", ctx.test)):
        print(f"{name:20s}: {df.image_id.nunique():6d} images, {len(df):7d} products")
