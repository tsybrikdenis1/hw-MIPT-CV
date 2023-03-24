import sys, os.path, cv2, numpy as np


def autocontrast(img: np.ndarray, white_percent: float, black_percent: float) -> np.ndarray:

    height, width = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    imageSize = height * width 
    blackPixelsNumber = imageSize * black_percent
    whitePixelsNumber = imageSize * white_percent
    
    counter = 0 
    for i in range(256):
        counter += hist[i]
        if counter >= blackPixelsNumber:
            xMin = i 
            break
        
    counter = 0 
    for i in range(255, -1,  -1):
        counter += hist[i]
        if counter > whitePixelsNumber:
            xMax = i 
            break 
    
    scale = 255 / (xMax - xMin)
    offset = -xMin * scale
    
    table = np.arange(256) * scale + offset
    table[table < 0] = 0 
    table[table > 255] = 255
    table = np.float32(table)
    table = np.around(table)
    table = table.astype(np.uint8)
    
    return table[img[:, :]]


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
