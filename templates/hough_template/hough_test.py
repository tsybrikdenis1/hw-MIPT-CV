import os
import pytest
import cv2
import numpy as np
from profiler import (Profiler, MemoryUnit, TimeUnit)
from hough import (hough_transform, get_lines)
from collections import namedtuple


class TestHT:
    Result = namedtuple('HTResult', [
        'ht_space', 'img', 'rhos', 'thetas', 'n_rhos', 'n_thetas',
        'elapsed_time', 'allocated_memory']
    )

    @pytest.fixture(
        params=['lines_bw.png', 'lines_gr.png'],
        ids=['bw', 'gray'],
        scope="session"
    )
    def img(self, request):
        path = request.param
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        return img

    @pytest.fixture(
        params=[(22, 22), (180, 180)],
        ids=['22x22', '180x180'],
        scope="session"
    )
    def result(self, request, img):
        n_rhos, n_thetas = request.param[:2]

        img_copy = img.copy()
        p = Profiler(TimeUnit.ms, MemoryUnit.KiB)
        ht_space, rhos, thetas = hough_transform(img_copy, n_rhos, n_thetas)
        elapsed_time, allocated_memory = p.stop()

        assert (img_copy == img).all()
        return TestHT.Result(ht_space, img, rhos, thetas, n_rhos, n_thetas,
                             elapsed_time, allocated_memory)

    @pytest.fixture(scope='session')
    def peaks(self, result):
        ht_space = result.ht_space
        second_peak = np.partition(ht_space.flatten(), -2)[-2]
        return ht_space >= second_peak

    def test_sum(self, img):  # Также проверяет на использование градаций серого
        ht_space = hough_transform(img, 1, 1)[0]
        assert img.sum() == ht_space[0, 0]

    def test_rhos(self, result):
        rhos = result.rhos
        assert len(rhos) == result.n_rhos

        rho_diff = abs(rhos + rhos[::-1]).max()
        assert rho_diff < 1e-5

        rho_delta = rhos[:-1] - rhos[1:]
        rho_delta_diff = abs(rho_delta - rho_delta[0]).max()
        assert rho_delta_diff < 1e-5

    def test_thetas(self, result):
        thetas = result.thetas
        assert len(thetas) == result.n_thetas

        theta_delta = thetas[:-1] - thetas[1:]
        theta_delta_diff = abs(theta_delta - theta_delta[0]).max()
        assert theta_delta_diff < 1e-5

        theta_diff = abs(thetas[1:] + thetas[:0:-1]).max()
        assert theta_diff < 1e-5

    def test_ht_space(self, result):
        ht_space = result.ht_space
        assert ht_space.dtype == np.uint32
        assert ht_space.shape == (result.n_rhos, result.n_thetas)

    def test_peaks_count(self, peaks):
        assert peaks.sum() == 2

    def test_peaks_coordinates(self, peaks, result):
        rhos, thetas = result.rhos, result.thetas
        peak_idxs = list(zip(*peaks.nonzero()))

        peak0 = abs(rhos + 95).argmin(), 0
        peak1 = abs(rhos - 95).argmin(), len(thetas) // 2
        assert peak0 in peak_idxs
        assert peak1 in peak_idxs

    def test_memory(self, result):
        assert result.allocated_memory < 500  # KiB

    def test_time(self, result):
        assert result.elapsed_time < 150  # ms

    @pytest.mark.xfail(reason='Optional lower time limit, use --runxfail option')
    def test_time_optional(self, request, result):
        assert result.elapsed_time < 50  # ms


class TestLine:
    Resources = namedtuple(
         'LineResources', ['elapsed_time', 'allocated_memory']
    )

    def test_center(self):
        center = np.zeros((3, 3), dtype=np.uint32)
        center[1, 1] = 1
        lines = get_lines(center, 2, 1, 1)
        assert len(lines) == 1
        assert [1, 1] in lines

    def test_corner(self):
        corner = np.zeros((3, 3), dtype=np.uint32)
        corner[0, 2] = 1
        lines = get_lines(corner, 2, 1, 1)
        assert len(lines) == 1
        assert [0, 2] in lines

    def test_const(self):
        const = np.ones((5, 5), dtype=np.uint32)
        lines = get_lines(const, 2, 1, 1)
        assert len(lines) == 0

    def test_many(self):
        many = np.zeros((5, 5), dtype=np.uint32)
        many[2, 0] = many[2, 4] = 2
        many[2, 2] = 1
        lines = get_lines(many, 3, 2, 2)
        assert len(lines) == 2
        assert [2, 0] in lines and [2, 4] in lines

    def test_eyes(self):
        eyes = np.zeros((3, 3), dtype=np.uint32)
        eyes[1, 0] = eyes[1, 2] = 1
        lines = get_lines(eyes, 2, 2, 2)
        assert len(lines) == 1
        assert [1, 0] in lines or [1, 2] in lines

    def test_random(self):
        random = np.random.choice(25, 25, False).reshape(5, 5)
        idx = np.squeeze(np.unravel_index(
            random.argmax(keepdims=True), random.shape))
        lines = get_lines(random, 1, 2, 2)
        assert len(lines) == 1
        assert idx in lines

    @pytest.fixture()
    def resources(self):
        size = 1024
        n_lines = 5
        large = np.random.choice(
            size ** 2, size ** 2, True).reshape(size, size)

        p = Profiler(TimeUnit.ms, MemoryUnit.KiB)
        lines = get_lines(large, n_lines, 3, 3)
        elapsed_time, allocated_memory = p.stop()

        assert len(lines) <= 5
        return TestLine.Resources(elapsed_time, allocated_memory)

    def test_memory(self, resources):
        assert resources.allocated_memory < 200  # KiB

    def test_time(self, resources):
        assert resources.elapsed_time < 150  # ms
