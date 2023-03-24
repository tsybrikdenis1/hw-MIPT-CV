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
    height, width = img.shape
    diag_len = np.ceil(np.sqrt(height ** 2 + width ** 2))
    rhos = np.linspace(-diag_len, diag_len, num = n_rhos)
    thetas = np.linspace(-np.pi/2, np.pi/2 - np.pi/n_thetas, num = n_thetas)
    H = np.zeros((n_rhos, n_thetas), dtype=np.uint32)
    are_edges = img > 50
    y_idxs, x_idxs = np.nonzero(are_edges)
    print(len(y_idxs))
    for x, y in zip(x_idxs, y_idxs):
        for t_idx in range(len(thetas)):
            rho_ = diag_len + int(round(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx])))
            H[int(round(rho_/((2 * diag_len)/(n_rhos-1)))), t_idx] += img[y, x]
    return H, rhos, thetas


def get_lines(
        ht_map: np.ndarray,
        n_lines: int,
        min_rho_line_diff: int,
        min_theta_line_diff: int,
) -> np.ndarray:
    if np.all(ht_map == ht_map[0]):
        return np.array([])
    res_thetas = []
    res_ros = []
    res = []
    ht_map_flat = np.ravel(ht_map)
    ind = np.argsort(ht_map_flat, axis=None)[::-1]
    H1_idx = np.unravel_index(ind, ht_map.shape)
    indx_t = 0 
    indx_r = 0 
    while len(res_thetas) < n_lines: 
        theta = H1_idx[0][indx_t]
        rho = H1_idx[1][indx_r]
        if not res_thetas and not res_ros:
            res_thetas.append(theta)
            res_ros.append(rho)
        if (min(np.abs(np.array(res_thetas) - theta)) > min_theta_line_diff
            or min(np.abs(np.array(res_ros) - rho)) > min_rho_line_diff) and ht_map[theta,rho] > 0:
            res_thetas.append(theta)
            res_ros.append(rho)
        if indx_t + 1 >= len(H1_idx[0]): 
            break
        indx_t += 1 
        indx_r += 1 
    for rho, theta in zip(res_ros, res_thetas):
        res.append([theta, rho])
    return np.array(res)


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, n_rhos, n_thetas, \
        n_lines, min_rho_line_diff, min_theta_line_diff = sys.argv[1:]

    n_rhos = int(n_rhos)
    assert n_rhos > 0

    n_thetas = int(n_thetas)
    assert n_thetas % 2 == 0 and n_thetas > 1

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
