import sys, os.path, cv2, numpy as np


def otsu(img: np.ndarray) -> np.ndarray:
    resultImage = img.copy()
    height, width = img.shape
    imageSize = width * height
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()/imageSize 
    
    left = 0
    right = 255
    while hist[left] == 0: 
        left += 1 
    while hist[right] == 0:
        right -= 1
        
    w_b, w_f = hist[left], 1 - hist[left]
    mu_b = left
    mu_f = np.sum(np.arange(left + 1, right + 1) * hist[left + 1 : right + 1]) / w_f
    sigma = w_b * w_f * (mu_b - mu_f) ** 2
    treshold = left
    
    for t in range(left + 1, right):
        p = hist[t]
        w_b_new, w_f_new = w_b + p, w_f - p

        mu_b_new = (mu_b * w_b + p * t) / w_b_new
        mu_f_new = (mu_f * w_f - p * t) / w_f_new

        w_b, w_f, mu_b, mu_f = w_b_new, w_f_new, mu_b_new, mu_f_new
        sigma_new = w_b * w_f * (mu_b - mu_f) ** 2

        if sigma_new > sigma:
            sigma, treshold = sigma_new, t   
            
    resultImage[resultImage > treshold] = 255
    resultImage[resultImage <= treshold] = 0
    return resultImage


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
