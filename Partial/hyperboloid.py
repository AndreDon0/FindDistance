import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Параметризация гиперболоида
def hyperboloid(u, v):
    x = np.cosh(u) * np.cos(v)
    y = np.cosh(u) * np.sin(v)
    z = np.sinh(u)
    return np.array([x, y, z])


# Первая фундаментальная форма
def fundamental_form(u):
    E = np.cosh(2 * u)
    G = np.cosh(u) ** 2
    return E, G


# Уравнения геодезических
def geodesic_eqs(s, vars):
    u, v, du_ds, dv_ds = vars
    E, G = fundamental_form(u)

    # Символы Кристоффеля
    Gamma_uuu = np.sinh(2 * u)
    Gamma_uuv = 0
    Gamma_uvv = -np.sinh(u) * np.cosh(u)
    Gamma_vuu = 0
    Gamma_vuv = np.tanh(u)
    Gamma_vvv = 0

    du2_ds2 = -Gamma_uuu * du_ds ** 2 - 2 * Gamma_uuv * du_ds * dv_ds - Gamma_uvv * dv_ds ** 2
    dv2_ds2 = -Gamma_vuu * du_ds ** 2 - 2 * Gamma_vuv * du_ds * dv_ds - Gamma_vvv * dv_ds ** 2

    return [du_ds, dv_ds, du2_ds2, dv2_ds2]


# Начальные условия
u0, v0 = 0, 0  # Начальная точка на гиперболоиде
u1, v1 = 1, np.pi / 2  # Конечная точка на гиперболоиде
du0_ds, dv0_ds = 0.1, 0.1  # Начальные скорости (приблизительные)

# Решение уравнений геодезических
s_span = [0, 10]
initial_conditions = [u0, v0, du0_ds, dv0_ds]

sol = solve_ivp(geodesic_eqs, s_span, initial_conditions, dense_output=True)

# Вычисление траектории
s_vals = np.linspace(0, 10, 500)
u_vals, v_vals = sol.sol(s_vals)[:2]

# Параметрические координаты в декартовы
xyz_vals = np.array([hyperboloid(u, v) for u, v in zip(u_vals, v_vals)])

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Поверхность гиперболоида
u = np.linspace(-2, 2, 100)
v = np.linspace(0, 2 * np.pi, 100)
u, v = np.meshgrid(u, v)
x = np.cosh(u) * np.cos(v)
y = np.cosh(u) * np.sin(v)
z = np.sinh(u)
ax.plot_surface(x, y, z, alpha=0.3, rstride=5, cstride=5)

# Траектория геодезической
ax.plot(xyz_vals[:, 0], xyz_vals[:, 1], xyz_vals[:, 2], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
