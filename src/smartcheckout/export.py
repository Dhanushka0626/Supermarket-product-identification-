"""
Export the trained model for local (CPU) use - the assignment requires local processing.

Owner: Member 2

export/
├── model.pt               YOLO weights (Modules B + C)
├── categories.json        17 class names (+ SKU -> class mapping for reference)
├── pipeline_config.json   tuned confidence threshold, NMS, preprocessing
└── metrics.json           final numbers (copy of outcomes_summary.json)
"""
from __future__ import annotations

import os
import shutil


def export_model(cfg: dict, paths, check_images=None) -> str:
    if not paths.best.exists():
        raise FileNotFoundError(f"trained model not found: {paths.best} - run scripts/train.py first")
    shutil.copy(paths.best, paths.export / "model.pt")
    summary = paths.outcomes / "outcomes_summary.json"
    if summary.exists():
        shutil.copy(summary, paths.export / "metrics.json")

    if check_images:                                   # reload on CPU exactly as on a laptop
        import time
        from .detection import load_pipeline
        from .preprocessing import load_image
        pipe = load_pipeline(paths.export, device="cpu")
        for p in check_images:
            t0 = time.time()
            _, d = pipe(load_image(p))
            print(f"CPU check - {os.path.basename(p)}: {len(d)} products in {time.time() - t0:.2f} s")

    (paths.export / "README.txt").write_text(
        "Unzip into the project's export/ folder, then run:\n"
        "  python demo.py --weights-dir export --images basket.jpg --out results\n")
    zip_path = shutil.make_archive(str(paths.work / "rpc17_export"), "zip", paths.export)
    print("Export ready:", zip_path, f"({os.path.getsize(zip_path) / 1e6:.0f} MB)")
    for f in sorted(paths.export.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:8.2f} MB")
    return zip_path
