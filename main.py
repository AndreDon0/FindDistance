import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

# Определим уравнение поверхности второго порядка
def surface_equation(x, y, A, B, C, D, E, F, G, H, I, J):
    return (-A*x**2 - B*y**2 - D*x*y - G*x - H*y - J) / C

# Определим функцию для расстояния между двумя точками
def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Задаем коэффициенты уравнения поверхности
A, B, C = 1, 1, -1
D, E, F = 0, 0, 0
G, H, I, J = 0, 0, 0, 0  # Пример: сфера x^2 + y^2 - z^2= 0

# Начальные точки (для примера)
initial_points = [0, 1, 0, 1, 0, 0]

# Определим функцию Лагранжа для минимизации
def lagrange_function(points, A, B, C, D, E, F, G, H, I, J):
    x1, y1, z1, x2, y2, z2 = points
    f1 = surface_equation(x1, y1, A, B, C, D, E, F, G, H, I, J) - z1
    f2 = surface_equation(x2, y2, A, B, C, D, E, F, G, H, I, J) - z2
    dist = distance(x1, y1, z1, x2, y2, z2)
    return dist + 1000 * (f1**2 + f2**2)  # Большой множитель для усиления ограничения

# Используем метод минимизации для нахождения минимального расстояния
result = minimize(lagrange_function, initial_points, args=(A, B, C, D, E, F, G, H, I, J), method='Nelder-Mead')

# Выводим результат
x1_min, y1_min, z1_min, x2_min, y2_min, z2_min = result.x
min_distance = distance(x1_min, y1_min, z1_min, x2_min, y2_min, z2_min)

print(f"Минимальное расстояние между точками ({x1_min}, {y1_min}, {z1_min}) и ({x2_min}, {y2_min}, {z2_min}) на поверхности: {min_distance}")

# Визуализация поверхности и точек
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Создаем сетку точек для визуализации поверхности
x = np.linspace(-1.5, 1.5, 400)
y = np.linspace(-1.5, 1.5, 400)
x, y = np.meshgrid(x, y)
z = surface_equation(x, y, A, B, C, D, E, F, G, H, I, J)

# Рисуем поверхность
ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100)

# Отмечаем точки с минимальным расстоянием
ax.scatter([x1_min, x2_min], [y1_min, y2_min], [z1_min, z2_min], color='red')
ax.text(x1_min, y1_min, z1_min, 'Point 1', color='red')
ax.text(x2_min, y2_min, z2_min, 'Point 2', color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface and Minimum Distance Points')

plt.show()
