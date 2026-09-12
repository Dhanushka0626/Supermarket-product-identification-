"""
Stage 3 — Product Classification  (Member B)

Loads the MobileNetV2 transfer-learning model trained by train_model.py and
classifies a single cropped product image into one of our category names.

Why MobileNetV2 with a frozen base: the assignment allows pretrained models
"if properly explained and integrated into the system pipeline." Freezing
the convolutional base means we reuse its general-purpose ImageNet visual
features as a fixed feature extractor; only the small head we add on top
(trained in train_model.py, on our own ~6-category dataset) does the actual
deciding. That's the line to point at in the viva.
"""

import json
from typing import List, Tuple

import cv2
import numpy as np

INPUT_SIZE = (160, 160)  # must match train_model.py


def preprocess_for_model(crop: np.ndarray, input_size=INPUT_SIZE) -> np.ndarray:
    """BGR crop (any size) -> normalized RGB float32 array ready for the model."""
    resized = cv2.resize(crop, input_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = rgb.astype("float32")
    # MobileNetV2's own preprocessing: scale to [-1, 1]
    array = (array / 127.5) - 1.0
    return array


def load_class_names(path: str) -> List[str]:
    with open(path, "r") as f:
        return json.load(f)


def load_classifier(model_path: str):
    """Lazy-imports TensorFlow so preprocessing/segmentation can be tested
    without it installed."""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def classify_crop(model, crop: np.ndarray, class_names: List[str]) -> Tuple[str, float]:
    """Returns (predicted_category, confidence in [0, 1])."""
    batch = np.expand_dims(preprocess_for_model(crop), axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])


def classify_crop_with_threshold(model, crop: np.ndarray, class_names: List[str],
                                  min_confidence: float = 0.5) -> Tuple[str, float]:
    """Same as classify_crop, but returns ("Other", confidence) below the
    threshold — useful when a detected object isn't one of our 6 trained
    categories (e.g. testing against real, uncurated checkout images)."""
    label, confidence = classify_crop(model, crop, class_names)
    if confidence < min_confidence:
        return "Other", confidence
    return label, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3 demo: classify one cropped product image.")
    parser.add_argument("image", help="Path to a cropped product image")
    parser.add_argument("--model", default="models/classifier.keras")
    parser.add_argument("--classes", default="models/class_names.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    crop = cv2.imread(args.image, cv2.IMREAD_COLOR)
    class_names = load_class_names(args.classes)
    model = load_classifier(args.model)
    label, confidence = classify_crop_with_threshold(model, crop, class_names, args.threshold)
    print(f"{args.image} -> {label} ({confidence:.1%} confidence)")
