"""
Stage 1 — Image Acquisition & Preprocessing  (Member A)

Cleans up a raw input image before segmentation:
resize -> denoise -> optional contrast fix -> color-space conversions.

Every function takes/returns a plain numpy BGR image (OpenCV's default),
so this module can be tested completely on its own with any product photo,
before segmentation or classification exist.
"""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR. Raises if the file can't be read."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resize_image(image: np.ndarray, max_dim: int = 900) -> np.ndarray:
    """Resize so the longer side is max_dim, keeping aspect ratio.

    Segmentation and classification both work more predictably (and faster)
    on a consistent scale than on whatever resolution the phone camera used.
    """
    h, w = image.shape[:2]
    scale = max_dim / float(max(h, w))
    if scale >= 1.0:
        return image  # don't upscale small images
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray, method: str = "median", ksize: int = 5) -> np.ndarray:
    """Remove sensor/compression noise before thresholding.

    - "median" is a good default: kills salt-and-pepper noise without
      smearing product edges as much as a plain Gaussian blur.
    - "gaussian" is smoother, useful if lighting is very uneven.
    """
    if method == "median":
        return cv2.medianBlur(image, ksize)
    if method == "gaussian":
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    raise ValueError(f"Unknown denoise method: {method}")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel (Lab space).

    Useful when a basket photo has uneven lighting (shadowed vs. lit products) —
    normal global histogram equalization would over-brighten already-lit areas.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """BGR -> HSV. Segmentation thresholds work better on Saturation/Value
    than on raw BGR because plain backgrounds are low-saturation."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def preprocess_image(path: str, max_dim: int = 900, denoise_method: str = "median",
                      use_clahe: bool = True) -> np.ndarray:
    """The full Stage 1 pipeline: load -> resize -> denoise -> (optional) CLAHE.

    Returns a cleaned BGR image, ready for segmentation.detect_products().
    """
    image = load_image(path)
    image = resize_image(image, max_dim=max_dim)
    image = denoise(image, method=denoise_method)
    if use_clahe:
        image = apply_clahe(image)
    return image


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Stage 1 demo: preprocess one image and save the result.")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--out", default="outputs/preprocessed.jpg", help="Where to save the cleaned image")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cleaned = preprocess_image(args.image)
    cv2.imwrite(args.out, cleaned)
    print(f"Saved preprocessed image to {args.out} ({cleaned.shape[1]}x{cleaned.shape[0]})")
