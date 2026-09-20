"""
Local demo of the Smart Supermarket Product Identification System (runs on CPU).   Owner: Member 2

Uses the exported model (download rpc17_export.zip from Kaggle and unzip it into export/):

    python demo.py --images samples/basket1.jpg samples/basket2.jpg
    python demo.py --images basket.jpg --conf 0.5 --denoise --clahe --out results

For every image: <name>_annotated.jpg (boxes + category labels), <name>_summary.csv (count and %
per category), <name>_chart.png (bar + pie chart), <name>_detections.json, plus a console summary.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import cv2  # noqa: E402

from smartcheckout.detection import load_pipeline  # noqa: E402
from smartcheckout.preprocessing import load_image  # noqa: E402
from smartcheckout.reporting.report import draw_detections, plot_report, print_report  # noqa: E402

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(items):
    out = []
    for it in items:
        p = Path(it)
        out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_TYPES) if p.is_dir() else [p]
    return out


def main():
    ap = argparse.ArgumentParser(description="Smart supermarket product identification (17 categories)")
    ap.add_argument("--weights-dir", default="export", help="folder with model.pt, categories.json, pipeline_config.json")
    ap.add_argument("--images", nargs="+", required=True, help="image files and/or folders")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=None, help="override the tuned confidence threshold")
    ap.add_argument("--denoise", action="store_true", help="Module A: median filter")
    ap.add_argument("--clahe", action="store_true", help="Module A: CLAHE contrast enhancement")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    pipe = load_pipeline(args.weights_dir, args.device)
    if args.conf is not None:
        pipe.cfg["conf"] = args.conf
    pp = {k: True for k in ("denoise", "clahe") if getattr(args, k)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images, all_dets = collect_images(args.images), []
    for p in images:
        t0 = time.time()
        proc, dets = pipe(load_image(p), preprocess=pp)
        dt = time.time() - t0
        cv2.imwrite(str(out / f"{p.stem}_annotated.jpg"), draw_detections(proc, dets, pipe.meta_names))
        df = print_report(dets, pipe.meta_names, title=f"{p.name}  ({dt:.2f} s)")
        df.to_csv(out / f"{p.stem}_summary.csv", index=False)
        plot_report(df, out / f"{p.stem}_chart.png", title=p.name)
        (out / f"{p.stem}_detections.json").write_text(json.dumps(dets, indent=2))
        all_dets += dets
    if len(images) > 1:
        df = print_report(all_dets, pipe.meta_names, title=f"ALL {len(images)} IMAGES")
        df.to_csv(out / "overall_summary.csv", index=False)
        plot_report(df, out / "overall_chart.png", title="All images")
    print(f"Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
