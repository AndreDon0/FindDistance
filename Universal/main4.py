import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial import distance
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

# Функция, задающая поверхность гиперболического параболоида
def f(x, y):
    return x**2 - y**2

# Задание параметров сетки для x и y
x_vals = np.linspace(-2, 2, 50)
y_vals = np.linspace(-2, 2, 50)

# Построение сетки точек для параметризованной поверхности
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)

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
p = [0.0, 0.0, f(0.0, 0.0)]
q = [1.0, -1.0, f(1.0, -1.0)]

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
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Сделать путь более жирным
path = np.array(path)
ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r', linewidth=3, label='Geodesic path')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parameterized Surface with Geodesic Path')
plt.legend()
plt.show()
