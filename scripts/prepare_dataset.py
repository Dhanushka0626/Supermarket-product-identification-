"""
Dataset prep — cuts the RPC (Retail Product Checkout) dataset's 200 fine-grained
SKU categories down to a handful of broad meta-categories we can actually
train a classifier on in a few hours.  (Member A)

RPC ships COCO-style annotation files (e.g. instances_train2019.json) with a
"categories" list like:
    {"id": 1, "name": "...", "supercategory": "puffed_food"}
"supercategory" is exactly the 17-group split the dataset's paper describes.
If your downloaded copy doesn't have that field under that name, run this
script with --inspect first, read off the real field/category names, and
either adjust CATEGORY_NAME_HINTS below or fill in MANUAL_CATEGORY_MAP by hand.

Usage
-----
    # 1. Look at what's actually in the file before filtering anything
    python scripts/prepare_dataset.py --annotations data/raw/instances_train2019.json --inspect

    # 2. Filter + subsample into data/train/<category>/
    python scripts/prepare_dataset.py \
        --annotations data/raw/instances_train2019.json \
        --images-dir data/raw/train2019 \
        --out data/train \
        --per-sku 120
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict


# The 6 categories this project classifies into. Edit this list (and the
# corresponding CATEGORY_NAME_HINTS below) if you swap categories after
# inspecting the real images.
CHOSEN_CATEGORIES = ["drink", "milk", "canned_food", "instant_noodles", "candy", "tissue"]

# Best-effort keyword fallback, only used if the annotation file has no
# "supercategory" field. Maps each chosen category to substrings that might
# appear in a raw category name. Extend this after running --inspect.
CATEGORY_NAME_HINTS = {
    "drink": ["drink", "cola", "juice", "water", "soda", "tea", "beverage"],
    "milk": ["milk", "dairy"],
    "canned_food": ["can", "canned", "tin"],
    "instant_noodles": ["noodle", "ramen"],
    "candy": ["candy", "sweet", "gum", "chocolate"],
    "tissue": ["tissue", "napkin", "paper"],
}

# If neither "supercategory" nor keyword matching gets you a clean split,
# hand-map specific category ids here: {category_id: "your_category_name"}
MANUAL_CATEGORY_MAP = {}


def load_categories(annotations_path: str):
    with open(annotations_path, "r") as f:
        coco = json.load(f)
    return coco.get("categories", []), coco.get("images", []), coco.get("annotations", [])


def inspect(annotations_path: str):
    categories, images, annotations = load_categories(annotations_path)
    print(f"{len(categories)} categories, {len(images)} images, {len(annotations)} annotations\n")
    has_supercategory = any("supercategory" in c for c in categories)
    print(f"'supercategory' field present: {has_supercategory}\n")
    print(f"{'id':>5}  {'name':<30}  supercategory")
    for c in categories[:40]:
        print(f"{c.get('id'):>5}  {str(c.get('name'))[:30]:<30}  {c.get('supercategory', '-')}")
    if len(categories) > 40:
        print(f"... and {len(categories) - 40} more (this is just a preview)")


def map_category_to_group(category: dict) -> str:
    """Decide which of CHOSEN_CATEGORIES (if any) this raw category belongs to."""
    cat_id = category.get("id")
    if cat_id in MANUAL_CATEGORY_MAP:
        return MANUAL_CATEGORY_MAP[cat_id]

    supercategory = str(category.get("supercategory", "")).lower()
    for group in CHOSEN_CATEGORIES:
        if group.replace("_", "") in supercategory.replace("_", "").replace(" ", ""):
            return group

    # Fallback: keyword match against the category name
    name = str(category.get("name", "")).lower()
    for group, hints in CATEGORY_NAME_HINTS.items():
        if group not in CHOSEN_CATEGORIES:
            continue
        if any(hint in name for hint in hints):
            return group

    return None  # not one of our chosen categories


def build_image_lookup(images: list) -> dict:
    return {img["id"]: img for img in images}


def prepare(annotations_path: str, images_dir: str, out_dir: str, per_sku: int, seed: int = 42):
    random.seed(seed)
    categories, images, annotations = load_categories(annotations_path)
    image_lookup = build_image_lookup(images)

    category_to_group = {c["id"]: map_category_to_group(c) for c in categories}
    matched = {cid: grp for cid, grp in category_to_group.items() if grp}
    print(f"Matched {len(matched)}/{len(categories)} raw categories to your {len(CHOSEN_CATEGORIES)} chosen groups.")
    if not matched:
        print("No categories matched — run with --inspect and fix CATEGORY_NAME_HINTS "
              "or MANUAL_CATEGORY_MAP before continuing.")
        return

    # Group annotations by (group, sku id) so we can subsample per-SKU, not
    # just per broad group — this keeps variety within each category instead
    # of accidentally training on 100 photos of the exact same product.
    by_sku = defaultdict(list)
    for ann in annotations:
        cat_id = ann.get("category_id")
        group = category_to_group.get(cat_id)
        if not group:
            continue
        img = image_lookup.get(ann.get("image_id"))
        if img is None:
            continue
        by_sku[(group, cat_id)].append(img["file_name"])

    os.makedirs(out_dir, exist_ok=True)
    totals = defaultdict(int)
    missing_files = 0

    for (group, cat_id), file_names in by_sku.items():
        file_names = list(dict.fromkeys(file_names))  # de-dupe, keep order
        random.shuffle(file_names)
        chosen = file_names[:per_sku]

        dest_dir = os.path.join(out_dir, group)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in chosen:
            src = os.path.join(images_dir, fname)
            if not os.path.exists(src):
                missing_files += 1
                continue
            dst = os.path.join(dest_dir, f"sku{cat_id}_{os.path.basename(fname)}")
            shutil.copyfile(src, dst)
            totals[group] += 1

    print("\nCopied images per category:")
    for group in CHOSEN_CATEGORIES:
        print(f"  {group:<18} {totals.get(group, 0)}")
    if missing_files:
        print(f"\n{missing_files} referenced files were not found under {images_dir} "
              f"(check --images-dir points at the right folder).")
    print(f"\nDone. Dataset ready at: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="Path to the COCO-style annotation json")
    parser.add_argument("--images-dir", help="Folder containing the actual image files (required unless --inspect)")
    parser.add_argument("--out", default="data/train", help="Output folder, one subfolder per category")
    parser.add_argument("--per-sku", type=int, default=120,
                         help="Max images to keep per fine-grained SKU (not per broad category)")
    parser.add_argument("--inspect", action="store_true", help="Print category info and exit, no files copied")
    args = parser.parse_args()

    if args.inspect:
        inspect(args.annotations)
    else:
        if not args.images_dir:
            parser.error("--images-dir is required unless you pass --inspect")
        prepare(args.annotations, args.images_dir, args.out, args.per_sku)
