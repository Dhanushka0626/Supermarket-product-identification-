"""Cross-validation summary tests.                                         Owner: Member 2"""
from smartcheckout.evaluation.cross_validation import CV_METRICS, summarize


def fake_fold(k, acc):
    base = {m: 0.5 for m in CV_METRICS}
    base.update(classification_acc=acc, ACD=1.0)
    return dict(fold=k, train_images=80, val_images=20, **base,
                by_level=[dict(level="all", classification_acc=acc, product_level_acc=0.5, count_acc=0.5,
                               cAcc=0.4, ACD=1.0)],
                ap50_per_class={"drink": 0.9, "candy": 0.8}, f1_per_class={"drink": 0.85, "candy": 0.7})


def test_mean_std_and_ci():
    out = summarize([fake_fold(k, a) for k, a in enumerate([0.80, 0.82, 0.84, 0.86, 0.88])],
                    ["drink", "candy"], final_test={"classification_acc": 0.9})
    row = out["summary"].loc["classification_acc"]
    assert abs(row["mean"] - 0.84) < 1e-9
    assert row["ci95_low"] < 0.84 < row["ci95_high"] <= 1.0
    assert row["final_model_test2019"] == 0.9
    assert list(out["per_class"].index) == ["drink", "candy"]


def test_no_results():
    assert summarize([], ["drink"]) is None
