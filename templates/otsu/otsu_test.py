import os
import time
import tracemalloc
import pytest
import cv2
import numpy as np
from otsu import otsu


class Profiler:
    def __init__(self):
        tracemalloc.start()
        self.start_time = time.time()

    def stop(self):
        elapsed_time = (time.time() - self.start_time) * 1000  # in ms
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        used_memory = float(peak - size) / 1024 / 1024  # in Mb
        return elapsed_time, used_memory


class TestOtsu:
    @pytest.fixture(params=[
        '120.80.10.npz',
        '160.80.10.npz',
        '160.80.20.npz',
        '200.60.30.npz',
        '200.60.40.npz',
        '200.60.40.large.npz'
    ])
    def img_data(self, request):
        test_name = request.param
        data = np.load(test_name)
        return data['img'], data['binarized'], data['threshold'], test_name

    @pytest.fixture()
    def result(self, img_data):
        p = Profiler()
        binarized = otsu(img_data[0].copy())
        resources = p.stop()

        assert binarized.dtype == np.uint8
        assert binarized.shape == img_data[1].shape
        return binarized, resources

    def test_binary(self, result):
        binarized = result[0]
        unique = np.unique(binarized)
        assert len(unique) == 2

    def test_unique(self, result, img_data):
        img = img_data[0]
        binarized = result[0]
        prev = -1
        for i in range(256):
            mask = img == i
            if not mask.any():
                continue
            unique = np.unique(binarized[mask])
            assert len(unique) == 1
            assert unique[0] >= prev
            prev = unique[0]

    def test_threshold_non_strict(self, result, img_data):
        img = img_data[0]
        thr = img_data[2]
        binarized = result[0]

        thr_upper = img[binarized > 0].min()
        thr_lower = img[binarized == 0].max()
        assert thr_lower < thr_upper
        assert abs(thr_lower - thr) <= 1

    def test_threshold_strict(self, result, img_data):
        img = img_data[0]
        thr = img_data[2]
        binarized = result[0]

        thr_upper = img[binarized > 0].min()
        thr_lower = img[binarized == 0].max()
        assert thr_lower < thr_upper
        assert thr_lower == thr

    def test_img(self, result, img_data):
        binarized = result[0]
        expected = img_data[1]
        assert (binarized == expected).all()

    def test_memory(self, result):
        used_memory = result[1][1]
        assert used_memory < 2  # Mb

    def test_time(self, result):
        elapsed_time = result[1][0]
        assert elapsed_time < 50  # ms
