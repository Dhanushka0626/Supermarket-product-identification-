"""
System-level evaluation metrics (shared by the test evaluation and cross-validation).

Owner: Member 2

A prediction "locates" a true product when IoU >= 0.5 (greedy matching, highest confidence first,
each true box used once, category ignored). On top of that:

* classification_acc  - located products that got the right category        (target >= 80 %)
* product_level_acc   - true products that were located AND correctly classified
* det_precision / det_recall - located / predicted, located / true
* count_acc           - images with the exact number of products
* cAcc                - images with the exact shopping list (count per category) - RPC paper metric
* ACD                 - average counting distance: sum over categories of |predicted - true| per image
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def make_groups(ann: pd.DataFrame) -> list:
    """One entry per image: (path, level, boxes xywh (N,4), classes (N,))."""
    return [(g.path.iloc[0], g.level.iloc[0], g[["x", "y", "w", "h"]].to_numpy(), g.meta.to_numpy())
            for _, g in ann.groupby("image_id")]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N,4) and b (M,4) in x1, y1, x2, y2 format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> list:
    iou = box_iou(pred, gt)
    pairs, used = [], set()
    for pi in np.argsort(-scores):
        cand = [(iou[pi, gi], gi) for gi in range(len(gt)) if gi not in used and iou[pi, gi] >= thr]
        if cand:
            _, gi = max(cand)
            used.add(gi)
            pairs.append((int(pi), int(gi)))
    return pairs


def score_image(dets: list, gt_xyxy: np.ndarray, gt_cls: np.ndarray) -> dict:
    """Compare the detections of one image with its ground truth."""
    pb = np.array([d["box"] for d in dets]).reshape(-1, 4)
    ps = np.array([d["meta_conf"] for d in dets])
    pm = [d["meta"] for d in dets]
    pairs = greedy_match(pb, ps, gt_xyxy)
    correct_p = {p for p, g in pairs if pm[p] == gt_cls[g]}
    correct_g = {g for p, g in pairs if pm[p] == gt_cls[g]}
    per_cls = Counter()
    for p, m in enumerate(pm):
        per_cls[(m, "tp" if p in correct_p else "fp")] += 1
    for g, m in enumerate(gt_cls):
        if g not in correct_g:
            per_cls[(int(m), "fn")] += 1
    a, b = Counter(pm), Counter(int(m) for m in gt_cls)
    return dict(
        record=dict(n_gt=len(gt_xyxy), n_pred=len(dets), located=len(pairs), correct=len(correct_p),
                    count_ok=len(dets) == len(gt_xyxy), cacc=a == b,
                    acd=sum(abs(a[k] - b[k]) for k in set(a) | set(b))),
        pairs=[(int(gt_cls[g]), pm[p]) for p, g in pairs],
        per_cls=per_cls)


class _ImageDataset:
    def __init__(self, groups):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        import cv2
        return i, cv2.imread(self.groups[i][0])


def _first(batch):
    return batch[0]


def evaluate_system(pipe, groups: list, n_classes: int, workers: int = 4, desc: str = "evaluation") -> dict:
    """Run the full system (Modules A -> B/C) on every image and score it against the ground truth."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    records, pair_true, pair_pred, t_total = [], [], [], 0.0
    per_cls = {c: Counter() for c in range(n_classes)}
    loader = DataLoader(_ImageDataset(groups), batch_size=1, num_workers=workers, collate_fn=_first)
    for i, img in tqdm(loader, total=len(groups), desc=desc):
        path, level, gxywh, gcls = groups[i]
        t0 = time.time()
        proc, dets = pipe(img)
        t_total += time.time() - t0
        s = proc.shape[1] / img.shape[1]                          # preprocessing may resize
        gt = np.c_[gxywh[:, :2], gxywh[:, :2] + gxywh[:, 2:]] * s
        res = score_image(dets, gt, gcls)
        records.append(dict(image=Path(path).name, level=level, **res["record"]))
        for t, p in res["pairs"]:
            pair_true.append(t); pair_pred.append(p)
        for (c, kind), n in res["per_cls"].items():
            per_cls[c][kind] += n
    return dict(rec=pd.DataFrame(records), pair_true=pair_true, pair_pred=pair_pred,
                per_cls={c: dict(v) for c, v in per_cls.items()},
                ms_per_image=1000 * t_total / max(1, len(groups)))


def aggregate(rec: pd.DataFrame) -> pd.Series:
    return pd.Series(dict(
        images=len(rec), products=int(rec.n_gt.sum()),
        classification_acc=rec.correct.sum() / max(1, rec.located.sum()),
        product_level_acc=rec.correct.sum() / max(1, rec.n_gt.sum()),
        det_precision=rec.located.sum() / max(1, rec.n_pred.sum()),
        det_recall=rec.located.sum() / max(1, rec.n_gt.sum()),
        count_acc=rec.count_ok.mean(), cAcc=rec.cacc.mean(), ACD=rec.acd.mean()))


def by_level(rec: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over all images and per difficulty level."""
    parts = [aggregate(rec).rename("all")] + [aggregate(g).rename(lv) for lv, g in rec.groupby("level")]
    out = pd.concat(parts, axis=1).T
    out.index.name = "level"
    return out


def per_class_prf(per_cls: dict, meta_names) -> pd.DataFrame:
    """Category-aware precision / recall / F1 for every class."""
    rows = []
    for c, v in per_cls.items():
        tp, fp, fn = v.get("tp", 0), v.get("fp", 0), v.get("fn", 0)
        rows.append(dict(category=meta_names[c], true_products=tp + fn,
                         precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn)))
    df = pd.DataFrame(rows)
    df["f1"] = (2 * df.precision * df.recall / (df.precision + df.recall).replace(0, np.nan)).fillna(0)
    return df.round(4)
