import sys, os.path, cv2, numpy as np


def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray, theta: float, rho: float
) -> (np.ndarray, list, list):
    pass  # insert your code here


def get_lines(
        ht_map: np.ndarray, n_lines: int,
        thetas: list, rhos: list,
        min_delta_rho: float, min_delta_theta: float
) -> list:
    pass  # insert your code here


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho, \
        n_lines, min_delta_rho, min_delta_theta = sys.argv[1:]

    theta = float(theta)
    assert theta > 0.0

    rho = float(rho)
    assert rho > 0.0

    n_lines = int(n_lines)
    assert n_lines > 0

    min_delta_rho = float(min_delta_rho)
    assert min_delta_rho > 0.0

    min_delta_theta = float(min_delta_theta)
    assert min_delta_theta > 0.0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    gradient = gradient_img(img.astype(float))

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(
        ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta
    )

    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write(f'{line[0]:.3f}, {line[1]:.3f}\n')


if __name__ == '__main__':
    main()
