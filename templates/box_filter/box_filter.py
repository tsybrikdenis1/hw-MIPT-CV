import sys, os.path, cv2, numpy as np


def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    pass  # implement your code here


def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    assert w > 0
    assert h > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = box_filter(img, w, h)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
