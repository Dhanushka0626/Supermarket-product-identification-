"""Phase 2 - train the final 17-class YOLO model (Modules B + C).          Owner: Member 1

Resumes automatically if a previous run was interrupted; skips if already trained.

    python scripts/train.py --config configs/config.yaml [--set epochs=60 batch=8]
"""
from _common import parse_args

from smartcheckout.data import dataset, yolo_format
from smartcheckout.training import train_final

if __name__ == "__main__":
    args, cfg, paths = parse_args(__doc__)
    ctx = dataset.load_context(cfg, paths)
    data_yaml = yolo_format.ensure_yolo_dataset(ctx, paths)
    print("Best weights:", train_final(cfg, paths, data_yaml))
