import sys, os.path, json, numpy as np
from numpy.random import normal
from scipy import stats
import matplotlib.pyplot as plt

def generate_data(
        img_size: tuple, line_params: tuple,
        n_points: int, sigma: float, inlier_ratio: float
) -> np.ndarray:
    w, h = img_size
    a, b, c = line_params
    inlier_number = round(n_points * inlier_ratio)
    outlier_number = n_points - inlier_number
    n_iterations = round(inlier_number * 1.3)
    X = []
    Y = []
    x_i = np.linspace(0, img_size[1]-1, inlier_number)
    indx = 0
    counter = 0
    while len(X) < inlier_number:
        assert counter < n_iterations
        noise = normal(loc=0, scale=sigma)
        y_i = (- line_params[2] - line_params[0] * x_i[indx])/ line_params[1]
        x_noise = x_i[indx] + noise
        y_noise = y_i + noise 
        if (x_noise > 0 and x_noise < img_size[1]) and (y_noise > 0 and y_noise < img_size[0]):
            X.append(x_noise)
            Y.append(y_noise)
            indx += 1
        counter += 1
    X_res = np.concatenate((np.array(X), np.random.uniform(0, w - 1, outlier_number)), axis=None)
    Y_res = np.concatenate((np.array(Y), np.random.uniform(0, h - 1, outlier_number)), axis=None)
    points = np.array((X_res, Y_res))
    return points

def compute_ransac_threshold(
        alpha: float, sigma: float
) -> float:
    # return np.square(stats.chi2.ppf(alpha, df = 2) * (sigma ** 2))
    return stats.chi2.ppf(alpha, sigma)


def compute_ransac_iter_count(
        conv_prob: float, inlier_ratio: float
) -> int:
    return  int(np.ceil(np.log(1 - conv_prob) / np.log(1 - inlier_ratio ** 2)))


def compute_line_ransac(
        data: np.ndarray, threshold: float, iter_count: int
) -> tuple:
    point_number = data.shape[1]
    best_score = 0 
    for i in range(iter_count): 
        indx = np.random.randint(0, point_number, size = 2)
        x_1 = data[0][indx[0]]
        x_2 = data[0][indx[1]]
        y_1 = data[1][indx[0]]
        y_2 = data[1][indx[1]]
        a, b, c = y_1 - y_2, x_2 - x_1, x_1 * y_2 - x_2 * y_1
        distance = np.abs(a * data[0] + b * data[1] + c) / np.sqrt(a **2 + b ** 2)
#         print(a, b, c)
        score = 0 
        for d in distance:
            if d < threshold:
                score += 1
        if score > best_score: 
            best_score, coeffs = score, (a, b, c)
    return coeffs

def show_points(data: np.ndarray, detected_line: tuple, 
                n_points: int, inlier_ratio: float, img_size: tuple):
    w, h = img_size
    inlier_number = round(n_points * inlier_ratio)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(data[0, :inlier_number],data[1, :inlier_number], label = 'inlier')
    ax.scatter(data[0, inlier_number:],data[1, inlier_number:], label = 'outlier')
    y_1 = (-detected_line[2])/ detected_line[1]
    y_2 = (-detected_line[2]-detected_line[0]*w)/ detected_line[1]
    plt.plot((0, w), (y_1, y_2), color = 'r', linewidth = 5, label = 'predicted line')
    ax.set_title('RANSAC')
    ax.legend()
    plt.show()
    
def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    )
    detected_line = compute_line_ransac(data, threshold, iter_count)
    show_points(data, detected_line, params['n_points'], params['inlier_ratio'], (params['w'], params['h']))
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()
