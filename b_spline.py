import numpy as np
import matplotlib.pyplot as plt

from TDMA import TDMASolve


def exact(x):
    return (x**2) / (x + 1)


def p(x):
    return -(x + 1)

def q(x):
    return 1


def f(x):
    return 2 * (1 - x - 2 * x**2 - x**3) / (x + 1)**3


def b_spline(t, i, xs, h, degree=3):
    if degree == 0:
        return np.where((xs[i] <= t) & (t < xs[i + 1]), 1, 0)
    res = ((t - xs[i]) / (degree * h) * b_spline(t, i, xs, h, degree - 1)
           + (xs[i + degree + 1] - t) / (degree * h) * b_spline(t, i + 1, xs, h, degree - 1))
    return res


def spline_sol(x, bs, xs, h):
    return sum(bs[i] * b_spline(x, i - 2, xs, h) for i in range(-1, n + 2))


a, b = 0, 1
alph_1, bet_1, gam_1 = 1, 0, 0
alph_2, bet_2, gam_2 = 0, 1, 0.75
n = 1000
xs = np.linspace(a, b, n + 1)
h = 1 / n
xs = np.append(xs, [b + i * h for i in range(1, 4)])
xs = np.append(xs, [a - (i) * h for i in range(3, 0, -1)])

A = np.zeros(n + 1)
D = np.zeros(n + 1)
C = np.zeros(n + 1)
F = np.zeros(n + 1)

for k in range(n + 1):
    A[k] = (1 - p(xs[k]) * h / 2 + q(xs[k]) * h ** 2 / 6) / (3 * h)
    D[k] = (1 + p(xs[k]) * h / 2 + q(xs[k]) * h ** 2 / 6) / (3 * h)
    C[k] = -A[k] - D[k] + q(xs[k]) * h / 3
    F[k] = f(xs[k]) * h / 3

Aa = alph_1 * h - 3 * bet_1
Ca = 4 * alph_1 * h
Da = alph_1 * h + 3 * bet_1
Fa = 6 * gam_1 * h
Ab = alph_2 * h - 3 * bet_2
Cb = 4 * alph_2 * h
Db = alph_2 * h + 3 * bet_2
Fb = 6 * gam_2 * h

C[0] -= Ca * A[0] / Aa
D[0] -= Da * A[0] / Aa
F[0] -= Fa * A[0] / Aa
A[-1] -= Ab * D[-1] / Db
C[-1] -= Cb * D[-1] / Db
F[-1] -= Fb * D[-1] / Db

bs = TDMASolve(A[1:], C, D[:-1], F)
ba = (Fa - bs[0] * Ca - bs[1] * Da) / Aa
bb = (Fb - bs[-1] * Cb - bs[-2] * Ab) / Db
bs = np.append(bs, [bb, ba])

N = 100000
x_sp = np.linspace(a - h * 3, b + h * 3, N)
x_ex = np.linspace(a, b, N)
u = spline_sol(x_sp, bs, xs, h)
exact_sol = exact(x_ex)

print(f'Максимальная ошибка: {max(abs(exact_sol - spline_sol(x_ex, bs, xs, h)))}')
plt.plot(x_sp, u, '--', label='В-сплайн')
plt.plot(x_ex, exact_sol, label='Точное решение')
plt.legend()
plt.grid()

plt.show()