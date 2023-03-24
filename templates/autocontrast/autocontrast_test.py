import os
import time
import tracemalloc
import pytest
import cv2
import numpy as np
from autocontrast import autocontrast


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


class TestAutocontrast:
    @pytest.fixture(scope="session")
    def img(self):
        img_path = 'auto_gray.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        return img

    @pytest.fixture(params=[
        (0.05, 32, 185, 'auto_gray_005_005_u.png', 'auto_gray_005_005_r.png'),
        (0.1, 56, 170, 'auto_gray_010_010_u.png', 'auto_gray_010_010_r.png'),
        (0.2, 88, 157, 'auto_gray_020_020_u.png', 'auto_gray_020_020_r.png'),
        (0.3, 106, 148, 'auto_gray_030_030_u.png', 'auto_gray_030_030_r.png')
    ], ids=[5, 10, 20, 30], scope="session")
    def result(self, request, img):
        ratio = request.param[0]

        p = Profiler()
        # копия, потому что кто-то может поменять аргумент внутри функции
        result = autocontrast(img.copy(), ratio, ratio)
        resources = p.stop()

        assert result.dtype == np.uint8
        assert result.shape == img.shape

        # if not is_correct_result:
        # output_img_path = os.path.join(output_dir_path, expected_u_name)
        # cv2.imwrite(output_img_path, result)

        return result, request.param, resources

    @pytest.fixture(scope="session")
    def expected_diff(self, result):
        result, params, _ = result
        expected_dir = f'expected/'
        expected_path = os.path.join(expected_dir, params[4])
        expected = cv2.imread(expected_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        assert expected is not None
        return abs(result - expected)

    def test_unique(self, img, result):
        result = result[0]
        prev = -1
        for i in range(256):
            mask = img == i
            if not mask.any():
                continue
            unique = np.unique(result[mask])
            assert len(unique) == 1
            assert unique[0] >= prev
            prev = unique[0]

    def test_black(self, img, result):
        result, params, _ = result
        t_result_black = img[result == 0].max()
        assert t_result_black == params[1]

    def test_white(self, img, result):
        result, params, _ = result
        t_result_white = img[result == 255].min()
        assert t_result_white == params[2]

    def test_non_strict(self, expected_diff):
        assert expected_diff.max() <= 1

    def test_strict(self, expected_diff):
        assert np.count_nonzero(expected_diff) == 0

    def test_memory(self, result):
        used_memory = result[2][1]
        assert used_memory < 1  # Mb

    def test_time(self, result):
        elapsed_time = result[2][0]
        assert elapsed_time < 50  # ms
