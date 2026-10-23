import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
from src.data.preprocessing import normalize_image, crop_volume


def test_normalize_image():
    img = Image.new('RGB', (100, 100), color=(128, 128, 128))
    normalized = normalize_image(img)
    assert normalized.shape == (100, 100, 3)
    assert normalized.dtype in [np.float32, np.float64]
    assert normalized.min() >= -3
    assert normalized.max() <= 3


def test_crop_volume():
    vol = np.random.rand(1, 128, 128, 128).astype(np.float32)
    cropped = crop_volume(vol, crop_size=(64, 64, 64))
    assert cropped.shape == (1, 64, 64, 64)
