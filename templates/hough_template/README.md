# Преобразование Хафа

На вход подается модуль градиента серого изображения (уже реализовано в шаблоне).
Требуется реализовать две функции: преобразование Хафа (ПХ) (1) и поиск прямых линий (2) с его помощью.

## 1. Преобразование Хафа

### Описание функции

Требуется перевести входное изображение **img** в ПХ размера **n_rhos** x **n_thetas**

### Сигнатура функции

``` python
def hough_transform(
        img: np.ndarray,
        n_rhos: int,
        n_thetas: int
) -> (np.ndarray, np.ndarray, np.ndarray)
```

### Параметры

- **img** - входное изображение (границы, полученные как модуль градиента)
- **n_thetas** - количество углов
- **n_rhos** - количество измерений по оси расстояния
- **ht_map** [out] - построенное ПХ; ht_map.shape = (n_rhos, n_thetas)
- **thetas** [out] - углы, для перевода из ПХ. len(thetas) = n_thetas; -pi/2 <= thetas[i] < pi/2 
- **rhos**  [out] - расстояния для перевода из ПХ. len(rhos) = n_rhos

## 2. Поиск прямых

### Описание функции

На вход функции подается ПХ (**ht_map**). По нему требуется найти **n_lines** наиболее выраженных прямых.  
Чтобы избежать множественного детектирования одной прямой, возвращаемые прямые должны отличаться не менее, 
чем на **min_rho_line_diff** по оси расстояний и **min_theta_line_diff** по оси углов.
В случае неразличимости прямых по данным параметрам, вернуть произвольную.

### Сигнатура функции

``` python
def get_lines(
        ht_map: np.ndarray,
        n_lines: int,
        min_rho_line_diff: int,
        min_theta_line_diff: int
) -> np.ndarray
```

### Параметры

- **ht_map** - ПХ
- **n_lines** - количество возвращаемых прямых
- **min_rho_line_diff** - минимальное расстояние между двумя ближайшими прямыми (целое число, в координатах ПХ (**ht_map**)) 
- **min_theta_line_diff** - минимальный угол между двумя ближайшими прямыми (целое число, в координатах ПХ (**ht_map**))
- **lines** [out] - список прямых (в координатах ПХ)