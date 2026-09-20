"""Evaluation metric tests.                                                Owner: Member 2"""
import numpy as np
import pandas as pd

from smartcheckout.evaluation.metrics import aggregate, box_iou, greedy_match, per_class_prf, score_image


def test_box_iou_known_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    np.testing.assert_allclose(box_iou(a, b)[0], [1.0, 50 / 150, 0.0], atol=1e-6)


def test_greedy_match_uses_each_true_box_once():
    gt = np.array([[0, 0, 10, 10]], float)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10]], float)
    assert greedy_match(pred, np.array([0.6, 0.9]), gt) == [(1, 0)]


def test_score_image_counts():
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], float)
    gt_cls = np.array([0, 1, 1])
    dets = [dict(box=[0, 0, 10, 10], meta=0, meta_conf=0.9),      # correct
            dict(box=[20, 0, 30, 10], meta=2, meta_conf=0.8),     # located, wrong class
            dict(box=[60, 0, 70, 10], meta=1, meta_conf=0.7)]     # false positive
    r = score_image(dets, gt, gt_cls)["record"]
    assert (r["n_gt"], r["n_pred"], r["located"], r["correct"]) == (3, 3, 2, 1)
    assert r["count_ok"] and not r["cacc"]
    assert r["acd"] == 2           # class 1: 1 predicted vs 2 true, class 2: 1 predicted vs 0 true


def test_aggregate_and_per_class():
    rec = pd.DataFrame([dict(n_gt=4, n_pred=4, located=4, correct=3, count_ok=True, cacc=False, acd=2),
                        dict(n_gt=2, n_pred=1, located=1, correct=1, count_ok=False, cacc=False, acd=1)])
    agg = aggregate(rec)
    assert agg.classification_acc == 4 / 5 and agg.product_level_acc == 4 / 6 and agg.count_acc == 0.5
    prf = per_class_prf({0: dict(tp=8, fp=2, fn=0), 1: dict(tp=0, fp=0, fn=3)}, ["a", "b"])
    assert prf.loc[0, "precision"] == 0.8 and prf.loc[0, "recall"] == 1.0 and prf.loc[1, "f1"] == 0
