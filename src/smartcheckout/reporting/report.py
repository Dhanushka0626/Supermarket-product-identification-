"""
Module D - statistics & report generation for one checkout image (or a batch of images).

Owner: Member 2

* total number of products, count and percentage per category
* console summary, CSV, bar + pie chart, image with labelled bounding boxes
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
import pandas as pd


def summarize(dets: list, meta_names) -> pd.DataFrame:
    """Category-wise counts and percentage distribution."""
    total = len(dets)
    c = Counter(d["meta_name"] for d in dets)
    rows = [dict(category=m, count=c[m], percent=round(100.0 * c[m] / total, 2)) for m in meta_names if c[m] > 0]
    df = pd.DataFrame(rows, columns=["category", "count", "percent"])
    return df.sort_values("count", ascending=False, ignore_index=True)


def print_report(dets: list, meta_names, title: str = "CHECKOUT SUMMARY") -> pd.DataFrame:
    df = summarize(dets, meta_names)
    line = "=" * 56
    print(line)
    print(title.center(56))
    print(line)
    print(f"Total products detected : {len(dets)}")
    print(f"Categories present      : {len(df)}")
    print("-" * 56)
    print(f"{'Category':<26}{'Count':>10}{'Percent':>14}")
    for cat, n, p in zip(df["category"], df["count"], df["percent"]):
        print(f"{cat:<26}{n:>10}{p:>13.1f}%")
    print(line)
    return df


def plot_report(summary_df: pd.DataFrame, save_path=None, show: bool = False, title: str = "Product distribution"):
    """Bar chart (counts) + pie chart (percentages)."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    if summary_df.empty:
        a1.text(0.5, 0.5, "No products detected", ha="center", va="center")
        a1.axis("off")
        a2.axis("off")
    else:
        cats, counts, pcts = summary_df["category"], summary_df["count"], summary_df["percent"]
        a1.barh(cats, counts, color="#4C78A8")
        a1.invert_yaxis()
        a1.set_xlabel("Count")
        a1.set_title("Products per category")
        for i, v in enumerate(counts):
            a1.text(v, i, f" {v}", va="center")
        wedges, _, _ = a2.pie(counts, autopct=lambda p: f"{p:.1f}%" if p >= 4 else "",
                              startangle=90, counterclock=False, pctdistance=0.75)
        a2.legend(wedges, [f"{c} ({p:.1f}%)" for c, p in zip(cats, pcts)],
                  loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        a2.set_title("Distribution (%)")
        a2.axis("equal")
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def palette(n: int) -> list:
    cols = []
    for i in range(n):
        hsv = np.uint8([[[int(180 * i / max(1, n)), 200, 235]]])
        cols.append(tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]))
    return cols


def draw_detections(img: np.ndarray, dets: list, meta_names) -> np.ndarray:
    """Bounding boxes with category labels (one colour per category) and the total count."""
    out = img.copy()
    pal = palette(len(meta_names))
    H, W = out.shape[:2]
    t = max(2, round(max(H, W) / 600))
    fs = max(0.5, max(H, W) / 1600)
    ft = max(1, t - 1)
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d["box"]]
        col = pal[d["meta"] % len(pal)]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, t)
        label = f"{d['meta_name']} {d.get('meta_conf', 1.0):.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        ty = max(y1, th + bl + 2)
        cv2.rectangle(out, (x1, ty - th - bl - 2), (x1 + tw + 4, ty), col, -1)
        cv2.putText(out, label, (x1 + 2, ty - bl), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft, cv2.LINE_AA)
    header = f"Total products: {len(dets)}"
    (tw, th), bl = cv2.getTextSize(header, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, ft + 1)
    cv2.rectangle(out, (0, 0), (tw + 16, th + bl + 16), (255, 255, 255), -1)
    cv2.putText(out, header, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.3, (0, 0, 0), ft + 1, cv2.LINE_AA)
    return out
