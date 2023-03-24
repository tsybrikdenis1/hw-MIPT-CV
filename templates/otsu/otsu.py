import sys, os.path, cv2, numpy as np


def otsu(img: np.ndarray) -> np.ndarray:
    pass  # implement your code here


def main():
    assert len(sys.argv) == 3
    src_path, dst_path = sys.argv[1], sys.argv[2]

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = otsu(img)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
