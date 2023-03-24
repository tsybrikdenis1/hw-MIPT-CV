import sys, os.path, cv2, numpy as np


def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    widowSize = w * h
    wl, wr = int(np.floor(w / 2)), int(np.floor((w-1) / 2))
    hl, hr = int(np.floor(h / 2)), int(np.floor((h-1) / 2))
    height = img.shape[0] - hl - hr 
    width = img.shape[1] - wl - wr
    int_image = cv2.integral(img)  
    blured = int_image[h:h+height, w:w+width] + int_image[:height, :width] - int_image[h:h+height, :width] - int_image[:height, w:w+width]
    blured = blured / widowSize
    blured = np.around(blured)
    return blured.astype('uint8')

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
