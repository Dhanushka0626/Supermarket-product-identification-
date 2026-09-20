"""Split tests: by image, stratified, reproducible.                        Owner: Member 1"""
import pandas as pd

from smartcheckout.data.splits import make_folds, make_holdout


def fake_ann(n_images=300):
    rows = []
    for i in range(n_images):
        level = ["easy", "medium", "hard"][i % 3]
        for _ in range(1 + i % 4):
            rows.append(dict(image_id=i, level=level, meta=i % 17))
    return pd.DataFrame(rows)


def test_holdout_size_and_stratification():
    ann = fake_ann()
    hold = make_holdout(ann, 0.10, seed=42)
    assert len(hold) == 30
    levels = ann.groupby("image_id")["level"].first()
    assert levels[list(hold)].value_counts().to_dict() == {"easy": 10, "medium": 10, "hard": 10}


def test_holdout_reproducible():
    ann = fake_ann()
    assert make_holdout(ann, 0.1, 1) == make_holdout(ann, 0.1, 1)


def test_folds_cover_all_images_once_and_are_balanced():
    ann = fake_ann()
    fold_of = make_folds(ann, 5, seed=42)
    assert set(fold_of) == set(ann.image_id.unique())
    sizes = pd.Series(fold_of).value_counts()
    assert sizes.max() - sizes.min() <= 3
