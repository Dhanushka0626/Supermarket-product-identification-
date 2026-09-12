"""
Stage 2 — Object Detection & Segmentation  (Member A)

Finds individual products in a preprocessed image using classical CV:
threshold -> morphology -> contours -> (optional) watershed split.

Works on two backgrounds the RPC dataset actually uses:
  - the plain turntable background behind single training-set products
  - the white checkout board behind multi-item basket/test images
Both are low-saturation, near-uniform backgrounds, which is exactly what
Otsu thresholding on the saturation channel is good at separating from
colorful product packaging.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Detection:
    """One detected product: its bounding box and the cropped pixels."""
    x: int
    y: int
    w: int
    h: int
    crop: np.ndarray   # BGR crop of just this product
    mask: np.ndarray   # binary mask (same size as crop) — 255 = product


def get_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask where 255 = likely product, 0 = background.

    Otsu's method picks the threshold automatically, which matters because
    lighting differs between the turntable shots and the checkout-board shots —
    a single hand-picked threshold wouldn't survive both.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological opening (remove speckle noise) then closing (fill holes)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def split_touching_objects(mask: np.ndarray) -> np.ndarray:
    """Watershed-based split for products that touch/overlap in the mask.

    Distance transform turns the mask into a 'how far from the edge' map;
    its local maxima become watershed markers (one per object), so two
    touching blobs get separated into two labeled regions instead of one.
    Returns an int32 label image (0 = background, 1..N = one object each).
    """
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, 0.5, 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    sure_bg = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1          # background becomes 1, not 0
    markers[unknown == 255] = 0    # unknown region is what watershed will fill in

    color_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color_image, markers)
    return markers


def find_product_contours(mask: np.ndarray, min_area: int = 800) -> List[np.ndarray]:
    """Contours of connected regions in the mask, filtered by minimum area
    to drop small speckles that survived cleaning."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def detect_products(image: np.ndarray, min_area: int = 800, use_watershed: bool = False,
                     padding: int = 4) -> List[Detection]:
    """Full Stage 2 pipeline: image -> list of Detection (bbox + crop + mask).

    Set use_watershed=True for cluttered basket images where products may
    touch; leave it False for the clean single-product training images
    (watershed there just adds noise for no benefit).
    """
    mask = get_foreground_mask(image)
    mask = clean_mask(mask)

    detections: List[Detection] = []
    h_img, w_img = mask.shape[:2]

    if use_watershed:
        labels = split_touching_objects(mask)
        for label in range(2, labels.max() + 1):  # 1 is background after +1 shift
            obj_mask = np.uint8(labels == label) * 255
            if cv2.countNonZero(obj_mask) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(obj_mask)
            x0, y0 = max(x - padding, 0), max(y - padding, 0)
            x1, y1 = min(x + w + padding, w_img), min(y + h + padding, h_img)
            detections.append(Detection(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                crop=image[y0:y1, x0:x1].copy(),
                mask=obj_mask[y0:y1, x0:x1].copy(),
            ))
    else:
        for contour in find_product_contours(mask, min_area=min_area):
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0 = max(x - padding, 0), max(y - padding, 0)
            x1, y1 = min(x + w + padding, w_img), min(y + h + padding, h_img)
            obj_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(obj_mask, [contour], -1, 255, thickness=cv2.FILLED)
            detections.append(Detection(
                x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                crop=image[y0:y1, x0:x1].copy(),
                mask=obj_mask[y0:y1, x0:x1].copy(),
            ))

    return detections


def visualize_boxes(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Debug view: draw unlabeled boxes only (no category yet — that's
    added later in stats.draw_annotated_image, after classification runs)."""
    out = image.copy()
    for det in detections:
        cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), (0, 200, 0), 2)
    return out


if __name__ == "__main__":
    import argparse
    import os
    from preprocessing import preprocess_image

    parser = argparse.ArgumentParser(description="Stage 2 demo: detect products in one image.")
    parser.add_argument("image", help="Path to an input image")
    parser.add_argument("--watershed", action="store_true", help="Split touching objects")
    parser.add_argument("--out", default="outputs/segmentation_debug.jpg")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cleaned = preprocess_image(args.image)
    dets = detect_products(cleaned, use_watershed=args.watershed)
    print(f"Detected {len(dets)} product(s).")
    debug_img = visualize_boxes(cleaned, dets)
    cv2.imwrite(args.out, debug_img)
    print(f"Saved debug visualization to {args.out}")
