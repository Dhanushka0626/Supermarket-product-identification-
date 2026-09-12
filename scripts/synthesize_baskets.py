"""
Synthetic basket-image generator.  (Joint — built on Member A's segmentation)

RPC's own multi-item "checkout" images mix in whatever's left of its other
~194 categories, so most won't match our 6-category classifier. Instead we
build our own "basket" test images: cut a clean product out of several
single-product training photos (easy, since those already have a plain
background) and composite them onto one canvas at random positions/rotations.

This gives unlimited multi-item test images with exactly-known ground truth
(useful for a live accuracy check in the demo), and is explicitly allowed by
the assignment brief ("manually captured images are acceptable").

Usage
-----
    python scripts/synthesize_baskets.py --source data/train --out data/synthetic_baskets --count 10
"""

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from segmentation import get_foreground_mask, clean_mask  # noqa: E402


def cutout_product(image_path: str):
    """Return (bgr_crop, alpha_mask) for the product in a single-product photo."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    mask = clean_mask(get_foreground_mask(image))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)
    crop = image[y:y + h, x:x + w]
    crop_mask = mask[y:y + h, x:x + w]
    return crop, crop_mask


def rotate_with_mask(crop, mask, angle):
    h, w = crop.shape[:2]
    diag = int(np.ceil(np.hypot(h, w)))
    center = (diag // 2, diag // 2)

    padded = np.zeros((diag, diag, 3), dtype=crop.dtype)
    padded_mask = np.zeros((diag, diag), dtype=mask.dtype)
    y0, x0 = (diag - h) // 2, (diag - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = crop
    padded_mask[y0:y0 + h, x0:x0 + w] = mask

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (diag, diag))
    rotated_mask = cv2.warpAffine(padded_mask, M, (diag, diag))
    return rotated, rotated_mask


def paste(canvas, crop, mask, x, y):
    h, w = crop.shape[:2]
    ch, cw = canvas.shape[:2]
    if y + h > ch or x + w > cw or x < 0 or y < 0:
        return False
    roi = canvas[y:y + h, x:x + w]
    mask3 = cv2.merge([mask, mask, mask]).astype(bool)
    roi[mask3] = crop[mask3]
    canvas[y:y + h, x:x + w] = roi
    return True


def synthesize_one(category_files: dict, canvas_size=(1000, 1000), items_per_image=(5, 9),
                    max_attempts=40):
    """category_files: {category_name: [image_path, ...]}"""
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 245, dtype=np.uint8)  # plain white-ish board
    n_items = random.randint(*items_per_image)
    placed = []

    categories = list(category_files.keys())
    for _ in range(n_items):
        category = random.choice(categories)
        if not category_files[category]:
            continue
        src_path = random.choice(category_files[category])
        crop, mask = cutout_product(src_path)
        if crop is None:
            continue

        scale = random.uniform(0.5, 0.9)
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
        mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]))
        angle = random.uniform(0, 360)
        crop, mask = rotate_with_mask(crop, mask, angle)

        h, w = crop.shape[:2]
        if h >= canvas_size[1] or w >= canvas_size[0]:
            continue

        for _ in range(max_attempts):
            x = random.randint(0, canvas_size[0] - w)
            y = random.randint(0, canvas_size[1] - h)
            if paste(canvas, crop, mask, x, y):
                placed.append({"category": category, "source": os.path.basename(src_path),
                               "bbox": [x, y, w, h]})
                break

    return canvas, placed


def main(source_dir: str, out_dir: str, count: int, seed: int = 7):
    random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    category_files = {}
    for category in sorted(os.listdir(source_dir)):
        cat_path = os.path.join(source_dir, category)
        if not os.path.isdir(cat_path):
            continue
        files = [os.path.join(cat_path, f) for f in os.listdir(cat_path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            category_files[category] = files

    if not category_files:
        print(f"No category folders with images found under {source_dir}. "
              f"Run scripts/prepare_dataset.py first.")
        return

    print(f"Compositing from {len(category_files)} categories: {list(category_files.keys())}")

    ground_truth = {}
    for i in range(count):
        canvas, placed = synthesize_one(category_files)
        out_name = f"basket_{i:03d}.jpg"
        cv2.imwrite(os.path.join(out_dir, out_name), canvas)
        ground_truth[out_name] = placed
        print(f"  {out_name}: {len(placed)} items placed")

    with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\nSaved {count} synthetic basket images + ground_truth.json to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/train", help="Folder of category subfolders (from prepare_dataset.py)")
    parser.add_argument("--out", default="data/synthetic_baskets")
    parser.add_argument("--count", type=int, default=10, help="How many basket images to generate")
    args = parser.parse_args()
    main(args.source, args.out, args.count)
