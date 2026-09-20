"""
Module A - image acquisition & preprocessing.

Owner: Member 1
"""
from __future__ import annotations

import cv2
import numpy as np


def load_image(path) -> np.ndarray:
    """Read an image from disk as BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def resize_max_side(img: np.ndarray, max_side: int | None) -> np.ndarray:
    """Shrink so the longest side is at most max_side (large phone photos -> training-like size)."""
    h, w = img.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def remove_noise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median filter: removes impulse / sensor noise while keeping edges sharp."""
    return cv2.medianBlur(img, ksize)


def enhance_contrast(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """CLAHE on the L channel (LAB colour space): evens out uneven lighting, keeps product colours."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(img: np.ndarray, max_side: int | None = 2000,
                     denoise: bool = False, clahe: bool = False) -> np.ndarray:
    """Full Module A chain: resize -> (optional) noise removal -> (optional) contrast enhancement."""
    out = resize_max_side(img, max_side)
    if denoise:
        out = remove_noise(out)
    if clahe:
        out = enhance_contrast(out)
    return out
