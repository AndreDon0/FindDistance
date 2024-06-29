import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial import distance
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

# Функция, задающая поверхность
def f(x, y, z):
    return np.log(x) + y**3 - 1/z + 4

# Задание параметров сетки для x и y
x_vals = np.linspace(0.1, 2, 50)  # x > 0 для функции ln(x)
y_vals = np.linspace(-1, 1, 50)

# Функция для численного решения уравнения ln(x) + y^3 - 1/z + 4 = 0
def solve_for_z(x, y):
    func = lambda z: f(x, y, z)
    z_initial_guess = 1.0  # Начальное приближение для решения
    try:
        z_solution, = fsolve(func, z_initial_guess, full_output=False)
        if not np.isfinite(z_solution):
            raise ValueError
    except (RuntimeError, ValueError):
        z_solution = np.nan  # Если решение не сходится, вернуть NaN
    return z_solution

# Построение сетки точек для параметризованной поверхности
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        Z[i, j] = solve_for_z(x, y)

# Удалить точки, где решение не удалось
valid_points = ~np.isnan(Z)
X = X[valid_points]
Y = Y[valid_points]
Z = Z[valid_points]

# Преобразование в плоский массив точек
points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

# Создание матрицы расстояний
dist_matrix = distance.cdist(points, points)

# Преобразование в разреженную матрицу
graph = csr_matrix(dist_matrix)

# Функция для нахождения ближайших индексов на сетке
def find_nearest_index(points, value):
    array = np.asarray(points)
    dist_2 = np.sum((array - value)**2, axis=1)
    return np.argmin(dist_2)

# Пример двух точек на поверхности
p = [0.5, 0.5, solve_for_z(0.5, 0.5)]
q = [1.5, -0.5, solve_for_z(1.5, -0.5)]

# Убедиться, что точки p и q не содержат NaN
if np.isnan(p).any() or np.isnan(q).any():
    raise ValueError("One of the points p or q contains NaN. Check if the point lies on the surface.")

# Поиск кратчайшего пути
start_index = find_nearest_index(points, p)
end_index = find_nearest_index(points, q)
distances, predecessors = shortest_path(csgraph=graph, directed=False, indices=start_index, return_predecessors=True)

# Восстановление пути
path = []
i = end_index
while i != start_index:
    path.append(points[i])
    i = predecessors[i]
    if i == -9999:
        raise ValueError("No path found between the points.")
path.append(points[start_index])
path = path[::-1]

# Вычисление длины пути
path = np.array(path)
geodesic_len = np.sum(np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1)))

print("Geodesic length:", geodesic_len)

# Визуализация параметризованной поверхности и геодезической линии
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c='b', marker='o', alpha=0.6)

ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r', label='Geodesic path')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parameterized Surface with Geodesic Path')
plt.legend()
plt.show()
