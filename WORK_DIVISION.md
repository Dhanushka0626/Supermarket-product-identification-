# Work division

Two members, each owning complete, explainable parts of the system. Every file starts with an
`Owner:` line, and `.github/CODEOWNERS` makes GitHub ask the owner to review changes to their files.

## Member 1 — data, preprocessing and the model (Modules A, B, C)

| File | Responsibility |
|---|---|
| `src/smartcheckout/config.py` | settings, shared paths, resuming a previous Kaggle session (shared file) |
| `src/smartcheckout/data/dataset.py` | load RPC annotations, **17-class definition from `supercategory`**, annotation tables |
| `src/smartcheckout/data/splits.py` | hold-out split, test subset, K-fold assignment (by image, stratified by level) |
| `src/smartcheckout/data/yolo_format.py` | YOLO label files, data YAML, fold file lists |
| `src/smartcheckout/preprocessing.py` | **Module A**: resize, median-filter noise removal, CLAHE |
| `src/smartcheckout/detection.py` | **Modules B + C**: YOLO inference, full pipeline, loading the exported model |
| `src/smartcheckout/training.py` | training the final model and the CV fold models (resume / skip logic, augmentation) |
| `scripts/prepare_data.py`, `scripts/train.py` | Phase 1 and Phase 2 entry points |
| `tests/test_preprocessing.py`, `tests/test_splits.py` | unit tests |

**Viva topics:** why `train2019` is not used (domain gap); how SKUs map to 17 classes; why splits are by image and
stratified; YOLO architecture (backbone, neck, detection head, anchor-free boxes, NMS); transfer learning from COCO;
augmentation choices (flips, HSV, mosaic, no rotation); image size 1024; median filter vs Gaussian; CLAHE in LAB space.

## Member 2 — evaluation, cross-validation and reporting (Module D)

| File | Responsibility |
|---|---|
| `src/smartcheckout/evaluation/metrics.py` | IoU, greedy matching, classification / product-level accuracy, cAcc, ACD, per-class P/R/F1 |
| `src/smartcheckout/evaluation/validation.py` | hold-out mAP, per-class AP, **confidence-threshold tuning** |
| `src/smartcheckout/evaluation/test_eval.py` | final evaluation on test2019 (cached) |
| `src/smartcheckout/evaluation/cross_validation.py` | **K-fold cross-validation**, mean ± std, 95 % CI |
| `src/smartcheckout/reporting/report.py` | **Module D**: counts, percentages, charts, labelled image |
| `src/smartcheckout/reporting/figures.py` | all report figures (confusion matrix, CV plots, examples) |
| `src/smartcheckout/reporting/outcomes.py` | outcomes package (markdown report, JSON, Excel, CSV, figures) |
| `src/smartcheckout/export.py`, `demo.py` | export for local CPU use, local demo |
| `scripts/validate.py`, `evaluate_test.py`, `cross_validate.py`, `build_outcomes.py`, `export_model.py` | Phase 3–6 entry points |
| `tests/test_metrics.py`, `tests/test_report.py`, `tests/test_cross_validation.py` | unit tests |

**Viva topics:** IoU and matching; precision / recall / F1 / mAP50 vs mAP50-95; why cAcc matters for a checkout;
how the threshold was chosen and why on the hold-out (not test); why test2019 is untouched; K-fold CV, why the
fold's last checkpoint is used (no selection bias), mean ± std and the t-based 95 % CI; confusion-matrix analysis.

## Shared

`README.md`, `configs/`, `scripts/run_all.py`, `notebooks/kaggle_runner.ipynb`, `.github/`.

## Interfaces between the two parts (agree on these first)

| Produced by Member 1 | Used by Member 2 |
|---|---|
| `load_context()` → `meta_names`, `train`, `hold`, `test` tables (`image_id, path, level, meta, x, y, w, h`) | evaluation, CV |
| `runs/rpc17/weights/best.pt` | validation, test, export |
| `Pipeline17(img) -> (image, [ {box, meta, meta_name, meta_conf}, ... ])` | metrics, report, demo |
| `train_fold(cfg, fold_dir, data_yaml, k)` → weights path | cross-validation |

| Produced by Member 2 | Used by Member 1 |
|---|---|
| `export/pipeline_config.json` (tuned threshold) | `detection.load_pipeline()` |

## GitHub workflow

1. **Member 1** creates the repository, adds Member 2 as collaborator, pushes the skeleton
   (`README.md`, `.gitignore`, `requirements*.txt`, `configs/`, `config.py`, package `__init__` files).
2. Protect `main` (Settings → Branches → require a pull request + 1 review).
3. Each member works on their own feature branches and opens pull requests; the **other member reviews**:

| Branch | Owner | Contents |
|---|---|---|
| `feature/data-preparation` | Member 1 | `data/`, `scripts/prepare_data.py`, `tests/test_splits.py` |
| `feature/preprocessing` | Member 1 | `preprocessing.py`, `tests/test_preprocessing.py` |
| `feature/model-training` | Member 1 | `detection.py`, `training.py`, `scripts/train.py` |
| `feature/evaluation-metrics` | Member 2 | `evaluation/metrics.py`, `validation.py`, `test_eval.py`, scripts, tests |
| `feature/report-module-d` | Member 2 | `reporting/report.py`, `figures.py`, `demo.py`, `tests/test_report.py` |
| `feature/cross-validation` | Member 2 | `evaluation/cross_validation.py`, `scripts/cross_validate.py`, tests |
| `feature/outcomes-export` | Member 2 | `reporting/outcomes.py`, `export.py`, scripts |
| `feature/kaggle-runner` | shared | `notebooks/`, `scripts/run_all.py`, README results |

4. Commit your own files from your own account in small, meaningful steps
   (e.g. `Add stratified hold-out split`, `Add greedy IoU matching and cAcc metric`), and make sure you can
   explain every line you commit.
5. After training, one member adds the final numbers to the README results table and tags `v1.0`.

```bash
git checkout -b feature/evaluation-metrics
git add src/smartcheckout/evaluation/metrics.py tests/test_metrics.py
git commit -m "Add IoU matching, classification accuracy, cAcc and ACD metrics"
git push -u origin feature/evaluation-metrics      # then open a pull request on GitHub
```
