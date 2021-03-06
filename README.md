## Преобразование Хафа

На вход подается модуль градиента серого изображения (уже реализовано в шаблоне).
Требуется реализовать 2 функции: преобразование Хафа (1) и поиск прямых линий (2) с его помощью.

## 1. Преобразование Хафа
 
``` python
def hough_transform(
        img: np.ndarray, theta: float, rho: float
) -> (np.ndarray, list, list)
```

Параметры:
- **img** - входное изображение (границы, полученные как модуль градиента)
- **theta** - шаг по оси углов (расстояние между двумя ближайшими углами в пространстве Хафа), в радианах
- **rho** - шаг по оси расстояния (аналогично **theta**, но в пикселях)
- **ht_map** [out] - построенное пространство Хафа; ht_map.shape = len(rhos), len(thetas)
- **thetas** [out] - ось углов
- **rhos**  [out] - ось расстояния

## 2. Поиск прямых

На вход функции подается посчитанное пространство Хафа (**ht_map**) и полученные значения **rhos** и **thetas**.

По нему требуется найти **n_lines** наиболее выраженных прямых, перевести в вид _y=kx+b_ и вернуть список параметров _(k_i, b_i)_.

Также требуется, чтобы в результате не возникало прямых, близких друг к другу. Для этого вводятся дополнительные параметры:
- **min_delta_rho** - минимальное расстояние между двумя ближайшими прямыми (в пикселях, как и **rho**)
- **min_delta_theta** - минимальный угол между двумя ближайшими прямыми (в радианах, как и  **theta**)


``` python
def get_lines(
        ht_map: np.ndarray, n_lines: int,
        thetas: list, rhos: list,
        min_delta_rho: float, min_delta_theta: float
) -> list
```
