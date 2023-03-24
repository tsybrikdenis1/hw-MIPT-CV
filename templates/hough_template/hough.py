import sys, os.path, cv2, numpy as np


def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray,
        n_rhos: int,
        n_thetas: int
) -> (np.ndarray, np.ndarray, np.ndarray):
    return  # implement your code here


def get_lines(
        ht_map: np.ndarray,
        n_lines: int,
        min_rho_line_diff: int,
        min_theta_line_diff: int,
) -> np.ndarray:
    return  # implement your code here


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, n_rhos, n_thetas, \
        n_lines, min_rho_line_diff, min_theta_line_diff = sys.argv[1:]

    n_rhos = int(n_rhos)
    assert n_rhos > 0

    n_thetas = int(n_thetas)
    assert n_thetas > 0

    n_lines = int(n_lines)
    assert n_lines > 0

    min_rho_line_diff = int(min_rho_line_diff)
    assert min_rho_line_diff > 0

    min_theta_line_diff = int(min_theta_line_diff)
    assert min_theta_line_diff > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    gradient = gradient_img(img.astype(np.float32))
    ht_map, rhos, thetas = hough_transform(img, n_rhos, n_thetas)

    dst_ht_map = ht_map.astype(np.float32)
    dst_ht_map /= dst_ht_map.max() / 255
    dst_ht_map = dst_ht_map.round().astype(np.uint8)
    cv2.imwrite(dst_ht_path, dst_ht_map)

    lines = get_lines(ht_map, n_lines, min_rho_line_diff, min_theta_line_diff)
    with open(dst_lines_path, 'w') as fout:
        for rho_idx, theta_idx in lines:
            fout.write(f'{rhos[rho_idx]:.3f}, {thetas[theta_idx]:.3f}\n')


if __name__ == '__main__':
    main()
