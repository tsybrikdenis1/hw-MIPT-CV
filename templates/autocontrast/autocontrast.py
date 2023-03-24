import sys, os.path, cv2, numpy as np


def autocontrast(img: np.ndarray, white_percent: float, black_percent: float) -> np.ndarray:
    pass  # Implement your code here


def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    white_percent, black_percent = float(sys.argv[3]), float(sys.argv[4])
    assert 0 <= white_percent < 1
    assert 0 <= black_percent < 1

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = autocontrast(img, white_percent, black_percent)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
