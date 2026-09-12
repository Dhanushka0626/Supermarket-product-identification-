"""
Stage 5 — Integration  (Joint)

The single command that ties every module together:
preprocess -> detect products -> classify each one -> aggregate & report.

Usage
-----
    python src/main.py --input data/synthetic_baskets --model models/classifier.keras \
        --classes models/class_names.json --output outputs/
"""

import argparse
import glob
import os

import cv2

from preprocessing import preprocess_image
from segmentation import detect_products
from classification import load_classifier, load_class_names, classify_crop_with_threshold
from stats import (compute_statistics, print_summary, draw_annotated_image,
                    plot_bar_chart, plot_pie_chart, save_results_json, save_results_csv)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def collect_images(input_path: str):
    if os.path.isdir(input_path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        return sorted(files)
    return [input_path]


def process_image(image_path: str, model, class_names, output_dir: str,
                   min_confidence: float, use_watershed: bool):
    name = os.path.splitext(os.path.basename(image_path))[0]

    cleaned = preprocess_image(image_path)
    raw_detections = detect_products(cleaned, use_watershed=use_watershed)

    results = []  # (x, y, w, h, label, confidence)
    for det in raw_detections:
        label, confidence = classify_crop_with_threshold(
            model, det.crop, class_names, min_confidence=min_confidence
        )
        results.append((det.x, det.y, det.w, det.h, label, confidence))

    labels = [label for (*_, label, _conf) in results]
    stats = compute_statistics(labels)
    print_summary(stats, image_name=os.path.basename(image_path))

    annotated = draw_annotated_image(cleaned, results)
    cv2.imwrite(os.path.join(output_dir, f"{name}_annotated.jpg"), annotated)

    if stats["total_products"] > 0:
        plot_bar_chart(stats, os.path.join(output_dir, f"{name}_bar.png"))
        plot_pie_chart(stats, os.path.join(output_dir, f"{name}_pie.png"))

    save_results_json(stats, results, os.path.join(output_dir, f"{name}_results.json"),
                       image_name=os.path.basename(image_path))
    save_results_csv(results, os.path.join(output_dir, f"{name}_detections.csv"))


def main():
    parser = argparse.ArgumentParser(description="Run the full basket-scan pipeline on an image or folder.")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--model", default="models/classifier.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                         help="Below this, a detection is labeled 'Other' instead of guessed")
    parser.add_argument("--watershed", action="store_true", default=True,
                         help="Split touching products (on by default — basket images are cluttered)")
    parser.add_argument("--no-watershed", dest="watershed", action="store_false")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading classifier...")
    model = load_classifier(args.model)
    class_names = load_class_names(args.classes)

    images = collect_images(args.input)
    if not images:
        print(f"No images found at {args.input}")
        return
    print(f"Found {len(images)} image(s) to process.\n")

    for image_path in images:
        process_image(image_path, model, class_names, args.output,
                       args.min_confidence, args.watershed)

    print(f"\nAll done. Annotated images, charts, and result files are in {args.output}/")


if __name__ == "__main__":
    main()
