import numpy as np


def canonical_form(matrix):
    # Разделим матрицу на квадратичную форму и линейные + свободные члены
    Q = matrix[:3, :3]  # Матрица квадратичной формы
    lin_terms = matrix[:3, 3]  # Линейные члены
    constant = matrix[3, 3]  # Свободный член

    # Найдём собственные значения и собственные векторы матрицы квадратичной формы
    eigenvalues, eigenvectors = np.linalg.eigh(Q)

    # Преобразование координат
    P = eigenvectors
    P_inv = np.linalg.inv(P)

    # Преобразование линейных членов
    new_lin_terms = P_inv @ lin_terms

    # Приведение к каноническому виду
    canonical_matrix = np.diag(eigenvalues)

    # Сдвиг координат для удаления линейных членов
    shift = -np.linalg.solve(canonical_matrix, new_lin_terms)

    # Каноническая форма без линейных членов
    canonical_form = np.zeros((4, 4))
    canonical_form[:3, :3] = canonical_matrix
    canonical_form[3, 3] = constant + new_lin_terms.T @ shift

    return canonical_form, shift


if __name__ == "__main__":
    # Пример использования
    # Матрица для уравнения Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
    example_matrix = np.array([
        [1, 0, 0, 0],  # A, D, E, G
        [0, 1, 0, 0],  # D, B, F, H
        [0, 0, 1, 0],  # E, F, C, I
        [0, 0, 0, 1]  # G, H, I, J
    ])


    canonical_matrix, shift = canonical_form(example_matrix)
    print("Каноническая форма матрицы:")
    print(canonical_matrix)
    print("Сдвиг координат:")
    print(shift)
