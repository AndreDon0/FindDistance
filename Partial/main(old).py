import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, diff, sin, cos, simplify, Matrix

# Шаг 1: Параметризация поверхности (сфера для примера)
u, v = symbols('u v')
x = sin(u) * cos(v)
y = sin(u) * sin(v)
z = cos(u)

# Вычисляем частные производные
r_u = [diff(x, u), diff(y, u), diff(z, u)]
r_v = [diff(x, v), diff(y, v), diff(z, v)]

# Шаг 2: Первая фундаментальная форма
E = sum([r_u[i] * r_u[i] for i in range(3)])
F = sum([r_u[i] * r_v[i] for i in range(3)])
G = sum([r_v[i] * r_v[i] for i in range(3)])

E = simplify(E)
F = simplify(F)
G = simplify(G)

print(f"E = {E}, F = {F}, G = {G}")

# Шаг 3: Символы Кристоффеля
g = Matrix([[E, F], [F, G]])
g_inv = g.inv()

Gamma = np.zeros((2, 2, 2), dtype=object)
coords = [u, v]

for i in range(2):
    for j in range(2):
        for k in range(2):
            Gamma[i, j, k] = 0.5 * sum([g_inv[i, m] * (diff(g[j, m], coords[k]) + diff(g[k, m], coords[j]) - diff(g[j, k], coords[m])) for m in range(2)])
            Gamma[i, j, k] = simplify(Gamma[i, j, k])

print(f"Gamma = {Gamma}")

# Шаг 4: Уравнения геодезических
def geodesic_eqs(s, vars):
    u_val, v_val, du_ds, dv_ds = vars
    du2_ds2 = -Gamma[0, 0, 0].subs({u: u_val, v: v_val}) * du_ds**2 \
              - 2 * Gamma[0, 0, 1].subs({u: u_val, v: v_val}) * du_ds * dv_ds \
              - Gamma[0, 1, 1].subs({u: u_val, v: v_val}) * dv_ds**2
    dv2_ds2 = -Gamma[1, 0, 0].subs({u: u_val, v: v_val}) * du_ds**2 \
              - 2 * Gamma[1, 0, 1].subs({u: u_val, v: v_val}) * du_ds * dv_ds \
              - Gamma[1, 1, 1].subs({u: u_val, v: v_val}) * dv_ds**2
    return [du_ds, dv_ds, du2_ds2, dv2_ds2]

# Шаг 5: Численное решение
phi0, theta0 = np.pi / 4, 0
dphi0_ds, dtheta0_ds = 0.1, 0.1
initial_conditions = [phi0, theta0, dphi0_ds, dtheta0_ds]

s_span = [0, 10]

sol = solve_ivp(geodesic_eqs, s_span, initial_conditions, dense_output=True)

s_vals = np.linspace(0, 10, 500)
u_vals, v_vals = sol.sol(s_vals)[:2]

# Параметризация сферы для декартовых координат
def sphere(u, v):
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    return np.array([x, y, z])

xyz_vals = np.array([sphere(u, v) for u, v in zip(u_vals, v_vals)])

# Шаг 6: Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
ax.plot_surface(x, y, z, alpha=0.3, rstride=5, cstride=5)

ax.plot(xyz_vals[:, 0], xyz_vals[:, 1], xyz_vals[:, 2], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
