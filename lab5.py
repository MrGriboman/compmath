# https://лови5.рф/upload/uf/09b/vraz7xspskir759cukwoqan1sedpcnt8/CHislennoe-reshenie.pdf
# quadrature method for Volterra eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.sin(s - x)


def f(x):
    return np.cos(x) + 0.125 * x ** 2 * np.cos(x) - 0.125 * x * np.sin(x)


def exact(x):
    return np.cos(x) - 0.5 * x * np.sin(x)


l = 1

a, b = 0, 3 * np.pi
n = 10

xs, h = np.linspace(a, b, n + 1, retstep=True)

F = f(xs)
w = l * np.ones(n + 1)
w[0] = w[-1] = l / 2

u = np.zeros(n + 1)

for i in range(n + 1):
    print(f"\r{i}", end="")
    Ki = lambda s: K(xs[i], s)
    u[i] = (F[i] + sum((w * Ki(xs) * h * u))) / (1 - w[i] * Ki(xs[i]))

e = exact(xs)

plt.figure("Решение")
plt.plot(xs, u)
plt.plot(xs, e, "--")
plt.legend(["Численное", "Точное"])
plt.title(f"{n = }")
plt.grid()

plt.figure("Модуль ошибки")
plt.plot(xs, abs(u - e))
plt.grid()

plt.show()