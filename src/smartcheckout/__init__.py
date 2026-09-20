"""Smart Supermarket Product Identification System (EC9570 Digital Image Processing).

Package layout (owner in brackets):
    config.py                  settings and shared paths                [Member 1, shared]
    data/                      dataset, splits, YOLO format             [Member 1]
    preprocessing.py           Module A - preprocessing                 [Member 1]
    detection.py               Modules B + C - YOLO inference pipeline  [Member 1]
    training.py                model training (final + CV folds)        [Member 1]
    evaluation/                metrics, validation, test, CV            [Member 2]
    reporting/                 Module D - report, figures, outcomes     [Member 2]
    export.py                  export for local use                     [Member 2]
"""
__version__ = "1.0.0"
