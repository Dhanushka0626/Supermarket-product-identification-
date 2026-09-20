"""
Train / hold-out / test / cross-validation splits.

Owner: Member 1

All splits are made **by image** (products from one photo never end up on both sides) and are
**stratified by difficulty level** (easy / medium / hard). They are saved to disk and reused, so
every run and every team member works with identical splits.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def image_levels(ann: pd.DataFrame) -> pd.Series:
    return ann.groupby("image_id")["level"].first()


def make_holdout(ann_val: pd.DataFrame, frac: float, seed: int) -> set:
    rng = np.random.default_rng(seed)
    levels = image_levels(ann_val)
    hold = set()
    for _, ids in levels.groupby(levels):
        ids = ids.index.to_numpy()
        hold |= set(rng.choice(ids, size=max(1, round(len(ids) * frac)), replace=False).tolist())
    return {int(i) for i in hold}


def load_or_make_holdout(paths, ann_val: pd.DataFrame, frac: float, seed: int) -> set:
    f = paths.data / "splits.json"
    if f.exists():
        return set(json.loads(f.read_text())["val_holdout_image_ids"])
    hold = make_holdout(ann_val, frac, seed)
    f.write_text(json.dumps({"val_holdout_image_ids": sorted(hold)}))
    return hold


def test_subset(ann_test: pd.DataFrame, limit, seed: int) -> pd.DataFrame:
    ids = ann_test.image_id.unique()
    if limit:
        ids = np.random.default_rng(seed).choice(ids, size=min(int(limit), len(ids)), replace=False)
    return ann_test[ann_test.image_id.isin(set(ids))].reset_index(drop=True)


def make_folds(ann_val: pd.DataFrame, k: int, seed: int) -> dict:
    """image_id -> fold number (0..k-1), stratified by difficulty level."""
    rng = np.random.default_rng(seed)
    levels = image_levels(ann_val)
    fold_of = {}
    for _, ids in levels.groupby(levels):
        for r, i in enumerate(rng.permutation(ids.index.to_numpy())):
            fold_of[int(i)] = r % k
    return fold_of


def load_or_make_folds(paths, ann_val: pd.DataFrame, k: int, seed: int) -> dict:
    f = paths.data / "folds.json"
    if f.exists():
        data = json.loads(f.read_text())
        if data.get("k") == k:
            return {int(i): v for i, v in data["fold_of"].items()}
    fold_of = make_folds(ann_val, k, seed)
    f.write_text(json.dumps({"k": k, "fold_of": fold_of}))
    return fold_of
