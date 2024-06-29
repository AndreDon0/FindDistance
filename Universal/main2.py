import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial import distance


# Функция, задающая поверхность
def f(x, y, z):
    return np.log(x) + y ** 3 - 1 / z + 4


# Задание параметров сетки для x и y
x_vals = np.linspace(0.1, 2, 100)  # x > 0 для функции ln(x)
y_vals = np.linspace(-1, 1, 100)


# Функция для численного решения уравнения ln(x) + y^3 - 1/z + 4 = 0
def solve_for_z(x, y):
    func = lambda z: f(x, y, z)
    z_initial_guess = 1.0  # Начальное приближение для решения
    z_solution, = fsolve(func, z_initial_guess)
    return z_solution


# Построение сетки точек для параметризованной поверхности
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        Z[i, j] = solve_for_z(x, y)

# Пример двух точек на поверхности
p = [0.5, 0.5, solve_for_z(0.5, 0.5)]
q = [1.5, -0.5, solve_for_z(1.5, -0.5)]


# Функция для вычисления длины геодезической линии между точками p и q
def geodesic_length(X, Y, Z, p, q):
    # Инициализация пути
    path = [p]
    current_point = np.array(p)
    end_point = np.array(q)

    while np.linalg.norm(current_point - end_point) > 0.01:
        distances = []
        indices = []

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j], Z[i, j]])
                dist = np.linalg.norm(point - current_point)
                distances.append(dist)
                indices.append((i, j))

        min_dist_index = np.argmin(distances)
        min_index = indices[min_dist_index]
        next_point = np.array([X[min_index], Y[min_index], Z[min_index]])

        path.append(next_point)
        current_point = next_point

    # Вычисление длины пути
    length = 0.0
    for i in range(1, len(path)):
        length += np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))

    return length


# Вычисление длины геодезической линии между p и q
geodesic_len = geodesic_length(X, Y, Z, p, q)
print("Geodesic length:", geodesic_len)

# Визуализация параметризованной поверхности и геодезической линии
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

path = np.array(geodesic_length(X, Y, Z, p, q, return_path=True))
ax.plot(path[:, 0], path[:, 1], path[:, 2], 'r', label='Geodesic path')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parameterized Surface with Geodesic Path')
plt.legend()
plt.show()
