import sys, os.path, cv2, math, numpy as np


def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray, theta: float, rho: float
) -> (np.ndarray, list, list):
    hw = img.shape

    # строим необходимые массивы и записываем их размеры
    rhoSize = int(round(hw[1]/rho) * round(hw[0]/rho))
    rho2 = round(hw[1]/rho)
    thetaSize = int(round(180 / theta))
    rhos = np.zeros(rhoSize)
    thetas = np.zeros(rhoSize)


    ht_map = np.zeros((rhoSize, thetaSize))
    for i in range(rhoSize):
        maxi, maxj, max = 0, 0, 0
        for k in range(int(rho * rho)):
            #ищем максимальный градиент в нашей окрестности и записываем его данные
            if(max < img[int(i // rho2 + k // rho)][int(i % rho2 + k % rho)]):
                maxi = i // rho2 + k // rho
                maxj = i % rho2 + k % rho
                max = img[int(i // rho2 + k // rho)][int(i % rho2 + k % rho)]
        rhos[i] = int(round(math.sqrt(maxi**2 + maxj**2)))
        thetas[i] = np.arctan(float(maxi)/maxj)

        for j in range(thetaSize):
            #записываем данные в таблицу среды Хафа
            A = (math.sin((theta/180) * j * math.pi))
            B = (math.cos((theta / 180) * j * math.pi))
            C = (maxj * B - maxi * A) * B
            ht_map[i][j] = abs(C)/(math.sqrt(A**2 + B**2))
    return(ht_map, thetas, rhos)

def get_lines(
        ht_map: np.ndarray, n_lines: int,
        thetas: list, rhos: list,
        min_delta_rho: float, min_delta_theta: float,theta
) -> list:

    out = np.zeros((n_lines, 2))
    for i in range(ht_map.shape[1]):
        R = 0
        KEKlist = np.zeros((ht_map.shape[0], 2))
        for j in range(ht_map.shape[0]):
            S = 0
            for k in range(j):
                if abs(ht_map[j][i] - KEKlist[k][0]) < min_delta_rho:
                    KEKlist[k][1] += 1
                    S += 1
            if(S == 0):
                KEKlist[R][1] = 1
                KEKlist[R][0] = ht_map[j][i]
                R += 1

    r = n_lines
    while r > 0:
        peepo = 0
        mi = 0
        mj = 0
        max = 0
        maxC = 0
        for i in range(ht_map.shape[1]):
            f = 0
            while f < ht_map.shape[0] and (KEKlist[f][1] != 0 or KEKlist[f][0] != 0):
                if(KEKlist[f][1] > max):
                    max = KEKlist[f][1]
                    maxC = KEKlist[f][0]
                    mi = i
                    mj = f
                f += 1
        for i in range(ht_map.shape[0]):
            for j in range(ht_map.shape[1]):
                if abs(ht_map[i][j] - maxC) < min_delta_rho and max > 0:
                    A = (math.sin((theta / 180) * mi * math.pi))
                    B = (math.cos((theta / 180) * mi * math.pi))
                    C = (rhos[i] * math.cos(thetas[i]) * B - rhos[i] * math.sin(thetas[i]) * A) * B
                    out[n_lines - r][0] = math.tan(((float(mi) * theta)/180) * math.pi)
                    out[n_lines - r][1] = C
                    peepo += 1
                    break
            if peepo > 0:
                break
        KEKlist[mj][1] = 0
        r -= 1
    return out


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
        ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta, theta
    )

    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write(f'{line[0]:.3f}, {line[1]:.3f}\n')


if __name__ == '__main__':
    main()