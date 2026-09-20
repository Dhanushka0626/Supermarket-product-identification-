"""Module A tests.                                                         Owner: Member 1"""
import numpy as np

from smartcheckout.preprocessing import enhance_contrast, preprocess_image, remove_noise, resize_max_side


def test_resize_caps_longest_side():
    img = np.zeros((3000, 4000, 3), np.uint8)
    out = resize_max_side(img, 2000)
    assert max(out.shape[:2]) == 2000 and out.shape[:2] == (1500, 2000)


def test_resize_keeps_small_images():
    img = np.zeros((500, 400, 3), np.uint8)
    assert resize_max_side(img, 2000) is img


def test_denoise_removes_salt_noise():
    img = np.full((50, 50, 3), 100, np.uint8)
    img[25, 25] = 255
    assert remove_noise(img)[25, 25, 0] == 100


def test_clahe_keeps_shape_and_increases_contrast():
    rng = np.random.default_rng(0)
    img = (100 + rng.integers(0, 20, (200, 200, 3))).astype(np.uint8)   # low-contrast image
    out = enhance_contrast(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.std() > img.std()


def test_full_chain():
    img = np.zeros((2400, 1200, 3), np.uint8)
    out = preprocess_image(img, max_side=1200, denoise=True, clahe=True)
    assert out.shape == (1200, 600, 3)
