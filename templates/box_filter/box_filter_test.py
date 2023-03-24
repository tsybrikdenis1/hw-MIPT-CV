import os
import time
import tracemalloc
import pytest
import cv2
import numpy as np
from box_filter import box_filter


class Profiler:
    def __init__(self):
        tracemalloc.start()
        self.start_time = time.time()

    def stop(self):
        elapsed_time = (time.time() - self.start_time) * 1000  # in ms
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        used_memory = float(peak - size) / 1000000  # in MB
        return elapsed_time, used_memory


class TestBoxFilter:
    @pytest.fixture(scope="session")
    def img(self):
        img_path = 'auto_gray.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        return img

    @pytest.fixture(params=[
        (2, 2), (3, 3), (5, 5), (7, 7), (9, 9), (10, 10), (11, 11),
        (51, 51), (101, 101), (201, 201)
    ], ids=[2, 3, 5, 7, 9, 10, 11, 51, 101, 201], scope="session")
    def result(self, request, img):
        w, h = request.param[:2]

        p = Profiler()
        actual = box_filter(img.copy(), w, h)
        resources = p.stop()

        # out_dirname = 'output'
        # if not os.path.exists(out_dirname):
        #     os.makedirs(out_dirname)
        # out_path = os.path.join(out_dirname, f'actual_{w}.png')
        # cv2.imwrite(out_path, actual)

        assert actual.dtype == np.uint8
        return actual, request.param, resources

    @pytest.fixture(scope="session")
    def expected(self, img, result):
        w, h = result[1][:2]
        expected = cv2.blur(
            img.astype(int), (w, h), borderType=cv2.BORDER_ISOLATED
        ).astype(np.float32)

        img_w, img_h = expected.shape
        wl, wr = int(np.floor(w / 2)), int(np.floor((w-1) / 2))
        hl, hr = int(np.floor(h / 2)), int(np.floor((h-1) / 2))
        return expected[wl:img_w-wr, hl:img_h-hr]

    def test_micro3(self):
        img = np.array([[1, 2], [3, 5]], dtype=np.uint8)
        actual = box_filter(img.copy(), 2, 2)
        assert 3 in actual

    def test_micro5(self):
        img = np.array([[1, 2], [3, 3]], dtype=np.uint8)
        actual = box_filter(img.copy(), 2, 2)
        assert 2 in actual

    def test_large(self):
        img = np.full((2000, 2000), 255, np.uint8)
        actual = box_filter(img.copy(), 2, 2)
        assert (actual[1:-1, 1:-1] == 255).all()

    def test_size_non_strict(self, result, expected):
        w_actual, h_actual = result[0].shape
        w_expected, h_expected = expected.shape
        assert w_actual >= w_expected - 1
        assert h_actual >= h_expected - 1

    def test_size_missed1(self, result, expected):
        w_actual, h_actual = result[0].shape
        w_expected, h_expected = expected.shape
        assert w_actual != w_expected - 1
        assert h_actual != h_expected - 1

    @pytest.fixture(scope="session")
    def corr(self, result, expected):
        actual = result[0].astype(np.float32)
        return cv2.matchTemplate(actual, expected, cv2.TM_CCORR_NORMED)

    def test_corr_pos(self, result, corr):
        actual_pos = np.asarray(np.unravel_index(corr.argmax(), corr.shape))
        expected_pos = np.asarray(corr.shape) // 2
        assert (actual_pos == expected_pos).all()

    def test_corr(self, result, expected, corr):
        actual = result[0].astype(np.float32)
        w = result[1][0]

        corr_max = corr.max()

        # if corr_max < 1:
        #     x0, y0 = np.unravel_index(corr.argmax(), corr.shape)
        #     wi, hi = expected.shape
        #     actual_center = actual[x0:x0 + wi, y0:y0 + hi]
        #     diff = abs(actual_center - expected)
        #     print(diff.max())
        #     diff[diff > 0] = 255
        #
        #     err_dir_name = 'errors'
        #     if not os.path.exists(err_dir_name):
        #         os.makedirs(err_dir_name)
        #     err_path = os.path.join(err_dir_name, f'diff_{w}.png')
        #     cv2.imwrite(err_path, diff)

        assert corr_max == 1.0

    def test_memory(self, result):
        used_memory = result[2][1]
        # w = result[1][0]
        # print(f'\nmemory used for filter of size {w}: {used_memory:0.2f} MB')
        assert used_memory < 10  # Mb

    def test_time(self, result):
        elapsed_time = result[2][0]
        # w = result[1][0]
        # print(f'\ntime for filter of size {w}: {elapsed_time:0.2f}')
        assert elapsed_time < 50  # ms
