import numpy as np
import scipy.linalg as la


def canonical_form_3d_with_shift(A, B, C, D, E, F, G, H, I, J):
    # Матрица коэффициентов квадратичной формы
    M = np.array([[A, D / 2, E / 2],
                  [D / 2, B, F / 2],
                  [E / 2, F / 2, C]])

    # Вектор линейных коэффициентов
    L = np.array([G, H, I])

    # Собственные значения и собственные векторы
    eigvals, eigvecs = la.eigh(M)

    # Каноническая форма
    A_c, B_c, C_c = eigvals

    # Матрица перехода (собственные векторы)
    P = eigvecs

    # Найдем новые координаты с учетом сдвига
    shift = -0.5 * la.solve(M, L)

    # Новые координаты: x' = P.T @ (x - shift)
    def new_coords(x, y, z):
        old_coords = np.array([x, y, z])
        return P.T @ (old_coords - shift)

    return A_c, B_c, C_c, P, shift, new_coords


# Пример
A, B, C, D, E, F, G, H, I, J = 3, 2, 4, 4, 2, 4, 5, 6, 7, 8

A_c, B_c, C_c, P, shift, new_coords = canonical_form_3d_with_shift(A, B, C, D, E, F, G, H, I, J)

print(f"Каноническая форма: {A_c}x'^2 + {B_c}y'^2 + {C_c}z'^2 = 0")
print("Матрица перехода (из новых координат в старые):")
print(P)
print("Сдвиг координат:")
print(shift)

# Пример использования функции для нахождения новых координат
x, y, z = 1, 2, 3
x_new, y_new, z_new = new_coords(x, y, z)
print(f"Новые координаты для точки ({x}, {y}, {z}): ({x_new}, {y_new}, {z_new})")
