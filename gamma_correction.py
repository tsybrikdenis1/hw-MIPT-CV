import sys, os.path, cv2, numpy as np

def gamma_correction(img: np.ndarray, a: float, b: float) -> np.ndarray:

    table = np.float32(np.arange(256))
    
    for i in range(1, 256): 
        table[i] = a * ((table[i] / 255.) ** b) * 255.
    
    table[table < 0] = 0 
    table[table > 255] = 255

    table = np.around(table)
    table = table.astype(np.uint8)
    
    return table[img[:, :]]

def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    a, b = float(sys.argv[3]), float(sys.argv[4])

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = gamma_correction(img, a, b)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
