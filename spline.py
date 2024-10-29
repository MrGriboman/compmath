import numpy as np
import matplotlib.pyplot as plt

from TDMA import TDMASolve


def exact(x):
    return (x**2) / (x + 1)


def p(x):
    return 0

def q(x):
    return 1


def f(x):
    return (2*x**2 / (x + 1)**3) - (4*x / (x + 1)**2) + ((2 + x**2) / (x + 1))


n = 10
alph_1, bet_1, gam_1 = 1, 0, 0
alph_2, bet_2, gam_2 = 0, 1, 0.75

xs= np.linspace(0, 1, n + 1)
h = 1 / n
mu = h / (h + h)
under = np.zeros(n + 1)
upper = np.zeros(n + 1)
main = np.zeros(n + 1)
F = np.zeros(n + 1)

main[0] = alph_1 * h - bet_1 * (1 - 1 / 3 * q(xs[0]) * h**2)
upper[0] = bet_1 * (1 + 1 / 6 * q(xs[1]) * h**2)
F[0] = gam_1 * h + 1 / 6 * bet_1 * h**2 * (2 * f(xs[0]) + f(xs[1]))

under[-1] = bet_2 * (-1 - 1 / 6 * h**2 * q(xs[-2]))
main[-1] = alph_2 * h + bet_2 * (1 - 1 / 3 * h**2 * q(xs[-1]))
F[-1] = gam_2 * h - 1 / 6 * bet_2 * h**2 * (f(xs[-2]) + 2 * f(xs[-1]))

for k in range(1, n):
    under[k] = (1 - mu) * (1 + h**2 * q(xs[k - 1]) / 6)
    main[k] = -(1 - h**2 * q(xs[k]) / 3)
    upper[k] = mu * (1 + h**2 * q(xs[k + 1]) / 6)
    F[k] = h**2 / 6 * (mu * f(xs[k - 1]) + 2 * f(xs[k]) + (1 - mu) * f(xs[k + 1]))


v = TDMASolve(under[1:], main, upper[:-1], F)
M = F - q(xs) * v

dots_between = 100
x_sol, u = np.empty(0), np.empty(0)

for i in range(n):
    x = np.linspace(xs[i], xs[i + 1], dots_between)

    t = (x - xs[i]) / h
    u_vi = v[i] * (1 - t) + v[i + 1] * t - h ** 2 / 6 * t * (1 - t) * ((2 - t) * M[i] + (1 + t) * M[i + 1])

    x_sol = np.append(x_sol, x)
    u = np.append(u, u_vi)


ex_sol = exact(x_sol)
print(f'Максимальная ошибка: {max(abs(u - ex_sol))}')

plt.plot(x_sol, u, "--", label='Сплайн')
plt.plot(x_sol, ex_sol, label='Точное решение')
plt.plot(xs, v, label='v')
plt.legend()
plt.grid(True)
plt.show()