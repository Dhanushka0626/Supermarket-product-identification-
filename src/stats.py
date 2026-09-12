"""
Stage 4 — Statistical Analysis & Report Generation  (Member B)

Turns a list of (bounding box, category, confidence) results for one image
into everything the assignment's "Statistical output requirements" section
asks for: total count, per-category counts, percentage distribution, a
chart, an annotated image, and an exportable result file.
"""

import csv
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe: no display needed to save PNGs
import matplotlib.pyplot as plt


def compute_statistics(labels: List[str]) -> Dict:
    """labels: one category name per detected product in the image."""
    total = len(labels)
    counts = Counter(labels)
    percentages = {cat: (count / total * 100 if total else 0) for cat, count in counts.items()}
    return {
        "total_products": total,
        "counts": dict(counts),
        "percentages": {k: round(v, 1) for k, v in percentages.items()},
    }


def print_summary(stats: Dict, image_name: str = ""):
    header = f"Summary for {image_name}" if image_name else "Summary"
    print(f"\n{header}")
    print("-" * len(header))
    print(f"Total products detected: {stats['total_products']}")
    print(f"{'Category':<20}{'Count':>8}{'Percent':>10}")
    for category, count in sorted(stats["counts"].items(), key=lambda kv: -kv[1]):
        pct = stats["percentages"][category]
        print(f"{category:<20}{count:>8}{pct:>9.1f}%")


def draw_annotated_image(image, detections: List[Tuple], color_map: Dict[str, tuple] = None):
    """detections: list of (x, y, w, h, label, confidence).

    Draws a bounding box + "label conf%" text per detection — the assignment's
    required "bounding boxes with labels" display mode.
    """
    out = image.copy()
    color_map = color_map or {}
    default_palette = [(0, 200, 0), (200, 120, 0), (0, 120, 220), (180, 0, 180),
                        (0, 180, 180), (200, 0, 60)]

    for i, (x, y, w, h, label, confidence) in enumerate(detections):
        color = color_map.get(label, default_palette[hash(label) % len(default_palette)])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 8)), (x + tw + 6, y), color, -1)
        cv2.putText(out, text, (x + 3, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def plot_bar_chart(stats: Dict, out_path: str, title: str = "Products per category"):
    categories = list(stats["counts"].keys())
    counts = [stats["counts"][c] for c in categories]
    plt.figure(figsize=(7, 4))
    plt.bar(categories, counts, color="#1B8A6B")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pie_chart(stats: Dict, out_path: str, title: str = "Category distribution"):
    categories = list(stats["percentages"].keys())
    values = [stats["percentages"][c] for c in categories]
    plt.figure(figsize=(5.5, 5.5))
    plt.pie(values, labels=categories, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_json(stats: Dict, detections: List[Tuple], out_path: str, image_name: str = ""):
    payload = {
        "image": image_name,
        "total_products": stats["total_products"],
        "counts": stats["counts"],
        "percentages": stats["percentages"],
        "detections": [
            {"bbox": [x, y, w, h], "category": label, "confidence": round(confidence, 3)}
            for (x, y, w, h, label, confidence) in detections
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_results_csv(detections: List[Tuple], out_path: str):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "w", "h", "category", "confidence"])
        for x, y, w, h, label, confidence in detections:
            writer.writerow([x, y, w, h, label, round(confidence, 3)])
