"""
Dataset loading and class definition (RPC dataset -> 17 meta-categories).

Owner: Member 1

* Locates the Kaggle copy of the RPC dataset and reads the COCO annotation files.
* Builds the class list from the dataset's own `supercategory` field (200 SKUs -> 17 classes).
* Builds one row per product (box) for val2019 and test2019 and saves the tables, so that
  every later step (training, evaluation, reporting) uses exactly the same data.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import splits

ANN_COLUMNS = ["image_id", "path", "width", "height", "level", "meta", "x", "y", "w", "h"]


def find_dataset_root(input_root: str | Path) -> Path:
    hits = sorted(glob.glob(os.path.join(str(input_root), "**", "instances_val2019.json"), recursive=True))
    if not hits:
        raise FileNotFoundError(
            f"RPC dataset not found under {input_root}. On Kaggle: '+ Add Input' -> "
            "diyer22/retail-product-checkout-dataset")
    return Path(hits[0]).parent


def load_coco(root: Path, splits_to_load=("val", "test")) -> dict:
    coco = {}
    for s in splits_to_load:
        with open(root / f"instances_{s}2019.json", encoding="utf-8") as f:
            coco[s] = json.load(f)
    return coco


def find_img_dir(root: Path, coco_split: dict, split: str) -> Path:
    sample = coco_split["images"][0]["file_name"]
    for c in (root / f"{split}2019", root / f"{split}2019" / f"{split}2019", root):
        if (c / sample).exists():
            return c
    for c in root.rglob(Path(sample).name):
        return c.parent
    raise FileNotFoundError(f"images for {split}2019 not found")


def meta_of(category: dict) -> str:
    """Meta-category of one SKU, taken from the dataset (never assigned by hand)."""
    return category.get("supercategory") or category["name"].split("_", 1)[-1]


def build_categories(coco: dict) -> dict:
    cats = sorted(coco["val"]["categories"], key=lambda c: c["id"])
    other = sorted(coco["test"]["categories"], key=lambda c: c["id"])
    if [(c["id"], c["name"]) for c in other] != [(c["id"], c["name"]) for c in cats]:
        raise ValueError("category list differs between val2019 and test2019")
    meta_names = []
    for c in cats:
        if meta_of(c) not in meta_names:
            meta_names.append(meta_of(c))
    return dict(meta_names=meta_names,
                catid_to_meta={c["id"]: meta_names.index(meta_of(c)) for c in cats},
                sku_to_meta={c["name"]: meta_of(c) for c in cats})


def class_definition_table(coco: dict) -> pd.DataFrame:
    cats = sorted(coco["val"]["categories"], key=lambda c: c["id"])
    hier = pd.DataFrame({"meta_category": [meta_of(c) for c in cats], "sku": [c["name"] for c in cats]})
    tab = hier.groupby("meta_category", sort=False).agg(
        skus_merged=("sku", "size"), example_skus=("sku", lambda s: ", ".join(s[:3])))
    tab.index.name = "class"
    return tab


def build_annotation_table(coco_split: dict, img_dir: Path, catid_to_meta: dict) -> pd.DataFrame:
    imgs = pd.DataFrame(coco_split["images"]).rename(columns={"id": "image_id"})
    if "level" not in imgs:
        imgs["level"] = "n/a"
    ann = pd.DataFrame(coco_split["annotations"])
    ann[["x", "y", "w", "h"]] = pd.DataFrame(ann["bbox"].tolist(), index=ann.index).astype(float)
    ann["meta"] = ann["category_id"].map(catid_to_meta).astype(int)
    df = ann.merge(imgs[["image_id", "file_name", "width", "height", "level"]], on="image_id")
    df["path"] = df["file_name"].map(lambda f: str(img_dir / f))
    df["level"] = df["level"].fillna("n/a").astype(str)
    return df[ANN_COLUMNS]


def prepare(cfg: dict, paths) -> SimpleNamespace:
    """Phase 1: read the dataset and save categories, annotation tables and splits."""
    root = find_dataset_root(cfg["input_root"])
    print("Dataset root:", root)
    coco = load_coco(root)
    cat = build_categories(coco)
    ann = {s: build_annotation_table(coco[s], find_img_dir(root, coco[s], s), cat["catid_to_meta"])
           for s in ("val", "test")}

    categories = dict(meta_names=cat["meta_names"], sku_to_meta=cat["sku_to_meta"])
    paths.categories.write_text(json.dumps(categories, indent=2, ensure_ascii=False), encoding="utf-8")
    for s, df in ann.items():
        df.to_csv(paths.data / f"ann_{s}.csv", index=False)
    class_definition_table(coco).to_csv(paths.results / "class_definition.csv")
    counts = pd.DataFrame({f"{s}2019": ann[s].meta.value_counts().reindex(range(len(cat["meta_names"])),
                                                                           fill_value=0).values
                           for s in ann}, index=cat["meta_names"])
    counts.index.name = "class"
    counts.to_csv(paths.results / "class_counts.csv")

    splits.load_or_make_holdout(paths, ann["val"], cfg["holdout_frac"], cfg["seed"])
    splits.load_or_make_folds(paths, ann["val"], cfg["cv_folds"], cfg["seed"])
    for s in ann:
        print(f"{s}2019: {ann[s].image_id.nunique()} images, {len(ann[s])} products, levels:",
              ann[s].groupby("image_id")["level"].first().value_counts().to_dict())
    print(f"{len(cat['sku_to_meta'])} SKUs merged into {len(cat['meta_names'])} classes")
    return load_context(cfg, paths)


def load_context(cfg: dict, paths) -> SimpleNamespace:
    """Everything later steps need, read back from the files written by prepare()."""
    if not (paths.data / "ann_val.csv").exists():
        return prepare(cfg, paths)
    meta_names = json.loads(paths.categories.read_text(encoding="utf-8"))["meta_names"]
    ann = {s: pd.read_csv(paths.data / f"ann_{s}.csv", dtype={"level": str}) for s in ("val", "test")}
    holdout_ids = splits.load_or_make_holdout(paths, ann["val"], cfg["holdout_frac"], cfg["seed"])
    test = splits.test_subset(ann["test"], cfg["test_limit"], cfg["seed"])
    return SimpleNamespace(
        meta_names=meta_names, n_classes=len(meta_names), ann=ann, holdout_ids=holdout_ids,
        train=ann["val"][~ann["val"].image_id.isin(holdout_ids)].reset_index(drop=True),
        hold=ann["val"][ann["val"].image_id.isin(holdout_ids)].reset_index(drop=True),
        test=test)
