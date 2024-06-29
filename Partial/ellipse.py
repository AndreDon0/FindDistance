import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Параметризация сферы
def sphere(phi, theta):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


# Первая фундаментальная форма
def fundamental_form(phi):
    E = 1
    G = np.sin(phi) ** 2
    return E, G


# Уравнения геодезических
def geodesic_eqs(s, vars):
    phi, theta, dphi_ds, dtheta_ds = vars


    # Символы Кристоффеля для сферы
    Gamma_phiphi_phi = 0
    Gamma_phitheta_theta = -np.sin(phi) * np.cos(phi)
    Gamma_thetaphi_phi = 0
    Gamma_thetatheta_phi = np.sin(phi) * np.cos(phi)
    Gamma_phiphi_theta = 0
    Gamma_phitheta_phi = 0
    Gamma_thetaphi_theta = 1 / np.sin(phi)
    Gamma_thetatheta_theta = 0

    dphi2_ds2 = -Gamma_phitheta_theta * dtheta_ds ** 2
    dtheta2_ds2 = -2 * Gamma_thetaphi_theta * dphi_ds * dtheta_ds

    return [dphi_ds, dtheta_ds, dphi2_ds2, dtheta2_ds2]


# Начальные условия
phi0, theta0 = np.pi / 4, 0  # Начальная точка на сфере
phi1, theta1 = np.pi / 3, np.pi / 2  # Конечная точка на сфере
dphi0_ds, dtheta0_ds = 0.1, 0.1  # Начальные скорости (приблизительные)

# Решение уравнений геодезических
s_span = [0, 10]
initial_conditions = [phi0, theta0, dphi0_ds, dtheta0_ds]

sol = solve_ivp(geodesic_eqs, s_span, initial_conditions, dense_output=True)

# Вычисление траектории
s_vals = np.linspace(0, 10, 500)
phi_vals, theta_vals = sol.sol(s_vals)[:2]

# Параметрические координаты в декартовы
xyz_vals = np.array([sphere(phi, theta) for phi, theta in zip(phi_vals, theta_vals)])

# Визуализация
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Поверхность сферы
phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
ax.plot_surface(x, y, z, alpha=0.3, rstride=5, cstride=5)

# Траектория геодезической
ax.plot(xyz_vals[:, 0], xyz_vals[:, 1], xyz_vals[:, 2], color='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
