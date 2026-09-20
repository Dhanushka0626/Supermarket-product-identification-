"""Module D tests.                                                         Owner: Member 2"""
import numpy as np

from smartcheckout.reporting.report import draw_detections, plot_report, summarize

NAMES = ["drink", "candy", "tissue"]
DETS = [dict(box=[10, 10, 60, 60], meta=0, meta_name="drink", meta_conf=0.9),
        dict(box=[70, 10, 120, 60], meta=0, meta_name="drink", meta_conf=0.8),
        dict(box=[10, 70, 60, 120], meta=1, meta_name="candy", meta_conf=0.7)]


def test_summarize_counts_and_percent():
    df = summarize(DETS, NAMES)
    assert df.category.tolist() == ["drink", "candy"]
    assert df["count"].tolist() == [2, 1]
    assert abs(df.percent.sum() - 100) < 0.1


def test_summarize_empty():
    assert summarize([], NAMES).empty


def test_draw_keeps_image_size(tmp_path):
    img = np.full((200, 200, 3), 255, np.uint8)
    out = draw_detections(img, DETS, NAMES)
    assert out.shape == img.shape and not np.array_equal(out, img)


def test_plot_report_writes_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    plot_report(summarize(DETS, NAMES), tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 1000
