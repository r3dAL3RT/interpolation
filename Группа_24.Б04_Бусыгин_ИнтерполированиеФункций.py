import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Метод Гаусса для решения системы линейных уравнений
# ------------------------------------------------------------
def gauss_elimination(A, b):
    """
    A - квадратная матрица (n x n), b - вектор (n,).
    Возвращает вектор x.
    """
    n = len(b)
    # Расширенная матрица [A | b]
    M = np.hstack((A.astype(float), b.reshape(-1, 1)))

    # Прямой ход
    for i in range(n):
        # Частичный выбор главного элемента в столбце i
        max_row = np.argmax(np.abs(M[i:, i])) + i
        if np.abs(M[max_row, i]) < 1e-15:
            raise ValueError("Матрица вырождена или близка к вырожденной")
        # Меняем строки местами
        M[[i, max_row]] = M[[max_row, i]]

        # Исключение переменной x_i из нижних строк
        for j in range(i + 1, n):
            factor = M[j, i] / M[i, i]
            M[j, i:] -= factor * M[i, i:]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1:n], x[i + 1:])) / M[i, i]

    return x

# ------------------------------------------------------------
# Исходная функция и интервал
# ------------------------------------------------------------
def f(x):
    return np.sqrt(x) + np.cos(x)

a, b = 0.1, 20

# ------------------------------------------------------------
# Узлы интерполяции (равноотстоящие и чебышёвские)
# ------------------------------------------------------------
def get_equidistant_nodes(n, a, b):
    """n равноотстоящих узлов на [a, b]"""
    return np.linspace(a, b, n)

def get_optimal_nodes(n, a, b):
    """n оптимальных узлов (Чебышёва) по формуле (3.2)"""
    i = np.arange(n)
    x_i = 0.5 * ((b - a) * np.cos((2*i + 1) * np.pi / (2*(n+1))) + (b + a))
    return x_i

# ------------------------------------------------------------
# Полином Лагранжа
# ------------------------------------------------------------
def lagrange_polynomial(x, nodes, values):
    n = len(nodes)
    result = 0.0
    for j in range(n):
        term = values[j]
        for k in range(n):
            if k != j:
                term *= (x - nodes[k]) / (nodes[j] - nodes[k])
        result += term
    return result

# ------------------------------------------------------------
# Полином Ньютона (разделённые разности + схема Горнера)
# ------------------------------------------------------------
def divided_differences(nodes, values):
    n = len(nodes)
    coef = np.zeros([n, n])
    coef[:, 0] = values
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (nodes[i+j] - nodes[i])
    return coef[0]

def newton_eval(x, nodes, coeffs):
    n = len(nodes)
    result = coeffs[n-1]
    for i in range(n-2, -1, -1):
        result = result * (x - nodes[i]) + coeffs[i]
    return result

# ------------------------------------------------------------
# Вычисление максимальных отклонений для полиномов
# ------------------------------------------------------------
def max_deviation(n_nodes, node_type='equidistant'):
    if node_type == 'equidistant':
        nodes = get_equidistant_nodes(n_nodes, a, b)
    else:
        nodes = get_optimal_nodes(n_nodes, a, b)

    values = f(nodes)
    coeffs = divided_differences(nodes, values)

    m = 1000                     # количество проверочных точек
    t = np.linspace(a, b, m)
    f_true = f(t)

    lagrange_vals = np.array([lagrange_polynomial(xi, nodes, values) for xi in t])
    newton_vals   = np.array([newton_eval(xi, nodes, coeffs) for xi in t])

    RL = np.max(np.abs(f_true - lagrange_vals))
    RN = np.max(np.abs(f_true - newton_vals))
    return RL, RN

# ------------------------------------------------------------
# СПЛАЙНЫ
# ------------------------------------------------------------
# ---- Линейный сплайн (S_{1,0}) ----
def linear_spline(x, nodes, values):
    i = np.searchsorted(nodes, x) - 1
    i = np.clip(i, 0, len(nodes)-2)
    x0, x1 = nodes[i], nodes[i+1]
    y0, y1 = values[i], values[i+1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

# ---- Квадратичный сплайн (S_{2,1}) ----
def quadratic_spline_coeffs(nodes, values):
    n = len(nodes)
    h = np.diff(nodes)
    a = values[:-1]
    b = np.zeros(n-1)
    c = np.zeros(n-1)
    b[0] = (values[1] - values[0]) / h[0]
    for i in range(n-1):
        c[i] = (values[i+1] - values[i] - b[i]*h[i]) / (h[i]**2)
        if i < n-2:
            b[i+1] = b[i] + 2*c[i]*h[i]
    return a, b, c

def quadratic_spline_eval(x, nodes, coeffs):
    a, b, c = coeffs
    i = np.searchsorted(nodes, x) - 1
    i = np.clip(i, 0, len(nodes)-2)
    dx = x - nodes[i]
    return a[i] + b[i]*dx + c[i]*dx**2

# ---- Кубический сплайн (S_{3,2}) ----
def cubic_spline_coeffs(nodes, values):
    n = len(nodes)
    h = np.diff(nodes)
    m = n - 2
    if m > 0:
        A = np.zeros((m, m))
        d_vec = np.zeros(m)
        for i in range(m):
            idx = i + 1
            lam = h[idx] / (h[idx-1] + h[idx])
            mu = h[idx-1] / (h[idx-1] + h[idx])
            d_vec[i] = 6/(h[idx-1] + h[idx]) * (
                (values[idx+1] - values[idx])/h[idx] - (values[idx] - values[idx-1])/h[idx-1]
            )
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = mu
            if i < m-1:
                A[i, i+1] = lam
        M_internal = gauss_elimination(A, d_vec)

        M = np.zeros(n)
        M[0] = 0.0
        M[-1] = 0.0
        M[1:-1] = M_internal
    else:
        M = np.zeros(n)

    a = values[:-1]
    b = np.zeros(n-1)
    c = np.zeros(n-1)
    d = np.zeros(n-1)
    for i in range(n-1):
        b[i] = (values[i+1] - values[i])/h[i] - h[i]*(2*M[i] + M[i+1])/6
        c[i] = M[i]/2
        d[i] = (M[i+1] - M[i])/(6*h[i])
    return a, b, c, d

def cubic_spline_eval(x, nodes, coeffs):
    a, b, c, d = coeffs
    i = np.searchsorted(nodes, x) - 1
    i = np.clip(i, 0, len(nodes)-2)
    dx = x - nodes[i]
    return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

# ---- Максимальное отклонение сплайна (равноотстоящие узлы) ----
def max_deviation_spline(n_nodes, spline_type):
    nodes = get_equidistant_nodes(n_nodes, a, b)
    values = f(nodes)
    if spline_type == 'linear':
        def spline_eval(x):
            return linear_spline(x, nodes, values)
    elif spline_type == 'quadratic':
        coeffs = quadratic_spline_coeffs(nodes, values)
        def spline_eval(x):
            return quadratic_spline_eval(x, nodes, coeffs)
    elif spline_type == 'cubic':
        coeffs = cubic_spline_coeffs(nodes, values)
        def spline_eval(x):
            return cubic_spline_eval(x, nodes, coeffs)
    else:
        raise ValueError("Unknown spline type")
    m = 1000
    t = np.linspace(a, b, m)
    f_true = f(t)
    spline_vals = np.array([spline_eval(xi) for xi in t])
    return np.max(np.abs(f_true - spline_vals))

# ------------------------------------------------------------
# ОСНОВНАЯ ПРОГРАММА
# ------------------------------------------------------------
if __name__ == "__main__":
    print("="*90)
    print("ИНТЕРПОЛЯЦИЯ ФУНКЦИИ: f(x) = √x + cos(x)")
    print(f"ИНТЕРВАЛ: [{a}, {b}]")
    print("="*90)

    n_values = [5, 10, 50, 100, 200]
    m = 1000

    # --------------------------------------------------------
    # ТАБЛИЦА 1: Полином Лагранжа
    # --------------------------------------------------------
    print("\nТАБЛИЦА №1: Полином Лагранжа")
    print("-"*40)
    print(f"{'n':<5} {'m':<8} {'RL_n (равноотст.)':<20} {'RLopt_n (оптим.)':<20}")
    print("-"*40)
    for n in n_values:
        RL_eq, _ = max_deviation(n, 'equidistant')
        RL_opt, _ = max_deviation(n, 'optimal')
        print(f"{n:<5} {m:<8} {RL_eq:<20.6e} {RL_opt:<20.6e}")

    # --------------------------------------------------------
    # ТАБЛИЦА 2: Полином Ньютона
    # --------------------------------------------------------
    print("\nТАБЛИЦА №2: Полином Ньютона")
    print("-"*40)
    print(f"{'n':<5} {'m':<8} {'RN_n (равноотст.)':<20} {'RNopt_n (оптим.)':<20}")
    print("-"*40)
    for n in n_values:
        _, RN_eq = max_deviation(n, 'equidistant')
        _, RN_opt = max_deviation(n, 'optimal')
        print(f"{n:<5} {m:<8} {RN_eq:<20.6e} {RN_opt:<20.6e}")

    # --------------------------------------------------------
    # ТАБЛИЦА 3: Сплайны (равноотстоящие узлы)
    # --------------------------------------------------------
    print("\nТАБЛИЦА №3: Интерполяционные сплайны (равноотстоящие узлы)")
    print("-"*90)
    print(f"{'n':<5} {'k':<8} {'RS_linear':<20} {'RS_quadratic':<20} {'RS_cubic':<20}")
    print("-"*90)
    for n in n_values:
        RS_lin = max_deviation_spline(n, 'linear')
        RS_quad = max_deviation_spline(n, 'quadratic')
        RS_cub = max_deviation_spline(n, 'cubic')
        print(f"{n:<5} {m:<8} {RS_lin:<20.6e} {RS_quad:<20.6e} {RS_cub:<20.6e}")

    # --------------------------------------------------------
    # ПОСТРОЕНИЕ ВСЕХ ГРАФИКОВ (все фигуры откроются одновременно)
    # --------------------------------------------------------
    x_plot = np.linspace(a, b, 1000)
    y_true = f(x_plot)
    colors = ['r', 'b', 'g', 'm', 'c']

    # ---------- Фигура 1: Лагранж с равноотстоящими узлами ----------
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='f(x)')
    for idx, n in enumerate(n_values):
        nodes = get_equidistant_nodes(n, a, b)
        values = f(nodes)
        y_interp = np.array([lagrange_polynomial(x, nodes, values) for x in x_plot])
        plt.plot(x_plot, y_interp, colors[idx] + '--', linewidth=1.5, label=f'L_{n}(x)')
        plt.plot(nodes, values, colors[idx] + 'o', markersize=4)
    plt.title('Полином Лагранжа с равноотстоящими узлами', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)

    # ---------- Фигура 2: Лагранж с оптимальными узлами ----------
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='f(x)')
    for idx, n in enumerate(n_values):
        nodes = get_optimal_nodes(n, a, b)
        values = f(nodes)
        y_interp = np.array([lagrange_polynomial(x, nodes, values) for x in x_plot])
        plt.plot(x_plot, y_interp, colors[idx] + '-', linewidth=1.5, label=f'Lopt_{n}(x)')
        plt.plot(nodes, values, colors[idx] + 'o', markersize=4)
    plt.title('Полином Лагранжа с оптимальными узлами (Чебышёва)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)

    # ---------- Фигура 3: Ньютон с равноотстоящими узлами ----------
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='f(x)')
    for idx, n in enumerate(n_values):
        nodes = get_equidistant_nodes(n, a, b)
        values = f(nodes)
        coeffs = divided_differences(nodes, values)
        y_interp = np.array([newton_eval(x, nodes, coeffs) for x in x_plot])
        plt.plot(x_plot, y_interp, colors[idx] + '--', linewidth=1.5, label=f'N_{n}(x)')
        plt.plot(nodes, values, colors[idx] + 'o', markersize=4)
    plt.title('Полином Ньютона с равноотстоящими узлами', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)

    # ---------- Фигура 4: Ньютон с оптимальными узлами ----------
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_true, 'k-', linewidth=2.5, label='f(x)')
    for idx, n in enumerate(n_values):
        nodes = get_optimal_nodes(n, a, b)
        values = f(nodes)
        coeffs = divided_differences(nodes, values)
        y_interp = np.array([newton_eval(x, nodes, coeffs) for x in x_plot])
        plt.plot(x_plot, y_interp, colors[idx] + '-', linewidth=1.5, label=f'Nopt_{n}(x)')
        plt.plot(nodes, values, colors[idx] + 'o', markersize=4)
    plt.title('Полином Ньютона с оптимальными узлами (Чебышёва)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8, loc='best')
    plt.grid(True, alpha=0.3)

    # ---------- Фигура 5: Демонстрация феномена Рунге ----------
    plt.figure(figsize=(14, 10))
    n_runge = 200

    plt.subplot(2, 2, 1)
    nodes_eq = get_equidistant_nodes(n_runge, a, b)
    values_eq = f(nodes_eq)
    coeffs_eq = divided_differences(nodes_eq, values_eq)
    y_interp_eq = np.array([newton_eval(x, nodes_eq, coeffs_eq) for x in x_plot])
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    plt.plot(x_plot, y_interp_eq, 'r--', linewidth=1.5, label=f'N_{n_runge}(x)')
    plt.plot(nodes_eq, values_eq, 'ro', markersize=4, label='Узлы')
    plt.title(f'ФЕНОМЕН РУНГЕ: Равноотстоящие узлы (n={n_runge})', fontsize=10, fontweight='bold', color='red')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    nodes_opt = get_optimal_nodes(n_runge, a, b)
    values_opt = f(nodes_opt)
    coeffs_opt = divided_differences(nodes_opt, values_opt)
    y_interp_opt = np.array([newton_eval(x, nodes_opt, coeffs_opt) for x in x_plot])
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    plt.plot(x_plot, y_interp_opt, 'g-', linewidth=1.5, label=f'Nopt_{n_runge}(x)')
    plt.plot(nodes_opt, values_opt, 'go', markersize=4, label='Узлы')
    plt.title(f'Оптимальные узлы Чебышёва (n={n_runge})', fontsize=10, fontweight='bold', color='green')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # ---------- Фигура 6: Сплайны и погрешности ----------
    plt.figure(figsize=(16, 12))
    n_subset = [5, 10, 20]
    colors_spline = ['r', 'b', 'g']

    # 6.1 Линейный сплайн
    plt.subplot(3, 2, 1)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    for idx, n in enumerate(n_subset):
        nodes = get_equidistant_nodes(n, a, b)
        values = f(nodes)
        y_spline = np.array([linear_spline(x, nodes, values) for x in x_plot])
        plt.plot(x_plot, y_spline, color=colors_spline[idx], linestyle='--', label=f'S1_0 n={n}')
        plt.plot(nodes, values, 'o', color=colors_spline[idx], markersize=4)
    plt.title('Линейный сплайн S_{1,0}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6.2 Квадратичный сплайн
    plt.subplot(3, 2, 2)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    for idx, n in enumerate(n_subset):
        nodes = get_equidistant_nodes(n, a, b)
        values = f(nodes)
        coeffs = quadratic_spline_coeffs(nodes, values)
        y_spline = np.array([quadratic_spline_eval(x, nodes, coeffs) for x in x_plot])
        plt.plot(x_plot, y_spline, color=colors_spline[idx], linestyle='--', label=f'S2_1 n={n}')
        plt.plot(nodes, values, 'o', color=colors_spline[idx], markersize=4)
    plt.title('Квадратичный сплайн S_{2,1}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6.3 Кубический сплайн
    plt.subplot(3, 2, 3)
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    for idx, n in enumerate(n_subset):
        nodes = get_equidistant_nodes(n, a, b)
        values = f(nodes)
        coeffs = cubic_spline_coeffs(nodes, values)
        y_spline = np.array([cubic_spline_eval(x, nodes, coeffs) for x in x_plot])
        plt.plot(x_plot, y_spline, color=colors_spline[idx], linestyle='--', label=f'S3_2 n={n}')
        plt.plot(nodes, values, 'o', color=colors_spline[idx], markersize=4)
    plt.title('Кубический сплайн S_{3,2}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6.4 Полиномы Лагранжа и Ньютона (n = 100) – для полноты оставим
    plt.subplot(3, 2, 4)
    n_poly = 100
    nodes = get_equidistant_nodes(n_poly, a, b)
    values = f(nodes)
    coeffs_newton = divided_differences(nodes, values)
    y_lagrange = np.array([lagrange_polynomial(x, nodes, values) for x in x_plot])
    y_newton = np.array([newton_eval(x, nodes, coeffs_newton) for x in x_plot])
    plt.plot(x_plot, y_true, 'k-', linewidth=2, label='f(x)')
    plt.plot(x_plot, y_lagrange, 'r--', label='Лагранж n=100')
    plt.plot(x_plot, y_newton, 'b:', label='Ньютон n=100')
    plt.plot(nodes, values, 'ko', markersize=4, label='узлы')
    plt.title('Полиномы Лагранжа и Ньютона (n=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6.5 Абсолютная погрешность (n = 100)
    plt.subplot(3, 2, 5)
    n_err = 100
    nodes = get_equidistant_nodes(n_err, a, b)
    values = f(nodes)
    y_lagrange = np.array([lagrange_polynomial(x, nodes, values) for x in x_plot])
    err_lagrange = np.abs(y_true - y_lagrange)
    coeffs_cubic = cubic_spline_coeffs(nodes, values)
    y_cubic = np.array([cubic_spline_eval(x, nodes, coeffs_cubic) for x in x_plot])
    err_cubic = np.abs(y_true - y_cubic)
    plt.semilogy(x_plot, err_lagrange, 'r-', label='Полином Лагранжа')
    plt.semilogy(x_plot, err_cubic, 'b--', label='Кубический сплайн')
    plt.title(f'Абсолютная погрешность, n = {n_err}')
    plt.ylabel('Погрешность')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6.6 Абсолютная погрешность (n = 100)
    plt.subplot(3, 2, 6)
    n_err = 100
    nodes = get_equidistant_nodes(n_err, a, b)
    values = f(nodes)
    y_lagrange = np.array([lagrange_polynomial(x, nodes, values) for x in x_plot])
    err_lagrange = np.abs(y_true - y_lagrange)
    coeffs_cubic = cubic_spline_coeffs(nodes, values)
    y_cubic = np.array([cubic_spline_eval(x, nodes, coeffs_cubic) for x in x_plot])
    err_cubic = np.abs(y_true - y_cubic)
    plt.semilogy(x_plot, err_lagrange, 'r-', label='Полином Лагранжа')
    plt.semilogy(x_plot, err_cubic, 'b--', label='Кубический сплайн')
    plt.title(f'Абсолютная погрешность, n = {n_err}')
    plt.ylabel('Погрешность')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
