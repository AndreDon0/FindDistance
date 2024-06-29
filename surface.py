import numpy as np

TYPES = {1: "Эллипсоид", 2: "Мнимый эллипсоид", 3: "Точка", 4: "Однополостный гиперболоид",
         5: "Двуполостный гиперболоид", 6: "Конус", 7: "Эллиптический параболоид",
         8: "Гиперболический параболоид", 9: "Эллиптический цилиндр",
         10: "Мнимый эллиптический цилиндр",
         11: "Прямая (пара мнимых пересекающихся плоскостей)",
         12: "Гиперболический цилиндр", 13: "Пара пересекающихся плоскостей",
         14: "Параболический цилиндр", 15: "Пара параллельных плоскостей",
         16: "Пара мнимых параллельных плоскостей", 17: "Плоскость"}

def raise_(ex):
    raise ex
def compute_minors(matrix, order):
    """
    Вычисляет все миноры заданного порядка для матрицы.
    """
    minors = []
    indices = list(range(matrix.shape[0]))
    from itertools import combinations

    for rows in combinations(indices, order):
        for cols in combinations(indices, order):
            submatrix = matrix[np.ix_(rows, cols)]
            minors.append(np.linalg.det(submatrix))

    return minors

class SurfaceSecondOrder:
    def __init__(self, B):
        self.B = B

        # Разделим матрицу на квадратичную форму и линейные + свободные члены
        self.A = B[:3, :3]  # Матрица квадратичной формы
        self.b = B[:3, 3]  # Линейные члены
        self.c = B[3, 3]  # Свободный член

        # Вычисление инвариантов
        self.I1 = np.trace(self.A)
        self.I2 = sum(compute_minors(self.A, 2))
        self.I3 = np.linalg.det(self.A)

        # Вычисление полуинвариантов
        self.K1 = np.linalg.det(self.A)
        self.K4 = np.linalg.det(self.B)

        if self.I2 == self.I3 == self.K4 == 0:
            self.K3 = sum(compute_minors(self.B, 3))
        else:
            self.K3 = lambda: raise_("Полуинвариант не работает")

        if self.I2 == self.I3 == self.K4 == self.K3 == 0:
            self.K2 = sum(compute_minors(self.B, 2))
        else:
            self.K2 = lambda: raise_("Полуинвариант не работает")

        self.type_surface = self.classify_surface()

        # Ищем каноническую форму и функции перехода
        self.canonical_form, self.to_new_cords, self.to_old_cords = self.to_canonical_form()

    # Определяем тип поверхности
    def classify_surface(self):
        if self.I3 != 0:
            if self.I2 > 0 and self.I1 * self.I3 > 0:
                if self.K4 > 0:
                    return 1
                elif self.K4 < 0:
                    return 2
                elif self.K4 == 0:
                    return 3
            elif self.I2 == 0 or self.I1 * self.I3 <= 0:
                if self.K4 < 0:
                    return 4
                elif self.K4 > 0:
                    return 5
                elif self.K4 == 0:
                    return 6
        if self.I3 == 0:
            if self.K4 < 0:
                return 7
            elif self.K4 > 0:
                return 8
            elif self.K4 == 0:
                if self.I2 > 0:
                    if self.I1 * self.K2 < 0:
                        return 9
                    elif self.I1 * self.K2 > 0:
                        return 10
                    elif self.K2 == 0:
                        return 11
                elif self.I2 < 0:
                    if self.K2 != 0:
                        return 12
                    elif self.K2 == 0:
                        return 13
                elif self.I2 == 0:
                    if self.K2 != 0:
                        return 14
                    elif self.K2 == 0:
                        if self.K1 < 0:
                            return 15
                        if self.K1 > 0:
                            return 16
                        if self.K1 == 0:
                            return 17
        else:
            exec("Поверхность не второго порядка")

    def to_canonical_form(self):

        # Найдём собственные значения и собственные векторы матрицы квадратичной формы
        eigenvalues, eigenvectors = np.linalg.eigh(self.A)

        # Преобразование координат
        P = eigenvectors
        P_inv = np.linalg.inv(P)

        # Преобразование линейных членов
        new_lin_terms = P_inv @ self.b

        # Приведение к каноническому виду
        canonical_matrix = np.diag(eigenvalues)

        # Сдвиг координат для удаления линейных членов
        shift = -np.linalg.solve(canonical_matrix, new_lin_terms)

        # Каноническая форма без линейных членов
        canonical_form = np.zeros((4, 4))
        canonical_form[:3, :3] = canonical_matrix
        canonical_form[3, 3] = self.c + new_lin_terms.T @ shift

        # Новые координаты: x' = P.T @ (x - shift)
        def new_cords(x, y, z):
            old= np.array([x, y, z])
            return P.T @ (old - shift)

        # Старые координаты: x = P @ x' + shift
        def old_cords(x, y, z):
            new = np.array([x, y, z])
            return P @ new + shift

        return canonical_form, new_cords, old_cords

    def parameterization(self, u, v):
        from math import sqrt
        from sympy import sin, cos, cosh, sinh
        A, B, C = self.canonical_form[0, 0], self.canonical_form[1, 1], self.canonical_form[2, 2]
        V = self.canonical_form[:3, 3]
        C = self.canonical_form[3, 3]
        a, b, c = sqrt(A), sqrt(B), sqrt(C)
        if self.type_surface == 3 or self.type_surface == 10 or self.type_surface == 16:
            exec(f"{TYPES[self.type_surface]} нельзя параметризовать")
        # x^2 +y^2 + z^2 = C
        elif self.type_surface == 1:
            x = a * sin(u) * cos(v)
            y = b * sin(u) * sin(v)
            z = c * cos(u)
        elif self.type_surface == 2:
            x, y, z = 0, 0, 0
        elif self.type_surface == 4:
            x = a * cosh(u) * cos(v)
            y = b * cosh(u) * sin(v)
            z = c * sinh(u)
        elif self.type_surface == 5:
            x = a * sinh(u) * cos(v)
            y = b * sinh(u) * sin(v)
            z = c * cosh(u)
        elif self.type_surface == 6:
            x = a * u * cos(v)
            y = b * u * sin(v)
            z = c * u
        elif self.type_surface == 7:
            p = 1 / V[2]
            x = a * u * cos(v)
            y = b * u * sin(v)
            z = p * u ** 2
        elif self.type_surface == 8:
            p = 1 / V[2]
            x = a * u
            y = b * v
            z = p * (u ** 2 - v ** 2)
        elif self.type_surface == 9:
            x = a * cos(u)
            y = b * sin(u)
            z = v
        elif self.type_surface == 11:
            ...
        elif self.type_surface == 12:
            x = a * cosh(u)
            y = b * sinh(u)
            z = v
        elif self.type_surface == 13:
            ...
        elif self.type_surface == 14:
            p = 1 / V[1]
            x = u
            y = p * u ** 2
            z = v
        elif self.type_surface == 15:
            ...
        elif self.type_surface == 17:
            a, b, c = V
            x = u
            y = v
            z = (C - a * u - b * v) / c

        return x, y, z

    def __repr__(self):
        return (f"Поверхность второго порядка"
                f"Тип: {TYPES[self.type_surface]}"
                f"Каноническая форма: {self.canonical_form}")


if __name__ == "__main__":
    # Пример использования
    B = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ])

    surface = SurfaceSecondOrder(B)
    surface_type = surface.classify_surface()
    print(f"Тип поверхности: {surface_type}")