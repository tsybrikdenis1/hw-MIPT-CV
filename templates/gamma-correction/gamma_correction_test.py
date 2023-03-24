import tracemalloc
import time
import cv2
import pytest
import numpy as np
from gamma_correction import gamma_correction


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


class TestGammaCorrection:
    params_1pix = np.load('params_1pix.npy', allow_pickle=True)
    params_4pix = np.load('params_4pix.npy', allow_pickle=True)

    @pytest.mark.parametrize("value, a, b, expected", params_1pix)
    def test_1pix(self, value, a, b, expected):
        result = gamma_correction(np.array(value, ndmin=2, dtype=np.uint8), a, b)
        assert result.dtype == np.uint8
        assert result.shape == (1, 1)
        assert result[0, 0] == expected

    @pytest.mark.parametrize("a, b, expected", params_4pix)
    def test_4pix(self, a, b, expected):
        img = np.array([[51, 102], [153, 204]], dtype=np.uint8)
        result = gamma_correction(img, a, b)
        assert result.dtype == np.uint8
        assert result.shape == (2, 2)
        assert (result == expected).all()

    @pytest.fixture(scope="session")
    def img(self):
        img_path = 'gamma_gray.png'
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        return img

    @pytest.fixture(params=[
        (2, 2, 'expected_2_2.png'),
        (0.5, 0.5, 'expected_05_05.png'),
    ], ids=[20, 5], scope="session")
    def result(self, request, img):
        a, b = request.param[:2]

        p = Profiler()
        # копия, потому что кто-то может поменять аргумент внутри функции
        result = gamma_correction(img.copy(), a, b)
        resources = p.stop()

        assert result.dtype == np.uint8
        assert result.shape == img.shape
        return result, request.param, resources

    @pytest.fixture(scope="session")
    def expected(self, result):
        _, params, _ = result
        expected = cv2.imread(params[2], cv2.IMREAD_GRAYSCALE)
        assert expected is not None
        return expected

    def test_img(self, result, expected):
        result, _, _ = result
        assert (result == expected).all()

    def test_memory(self, result):
        used_memory = result[2][1]
        with open('time.log', 'a') as f:
            f.write(f'memory: {used_memory}\n')
        assert used_memory < 1  # Mb

    def test_time(self, result):
        elapsed_time = result[2][0]
        with open('time.log', 'a') as f:
            f.write(f'time: {elapsed_time}\n')
        assert elapsed_time < 50  # ms
