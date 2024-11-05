# simple iteration for Volterra eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.sin(s - x)
    return np.sin(0.6 * x * s) / (s + 1e16)


def f(x):
    return np.cos(x) + 0.125 * x**2 * np.cos(x) - 0.125 * x * np.sin(x)
    return x


def exact(x):
    return np.cos(x) - 0.5 * x * np.sin(x)
    return 0*x




def solve(n=100, eps=1e-5, max_iter=10000, retstep=False):
    xl, h = np.linspace(a, b, n + 1, retstep=True)
    
    fx = f(xl)
    u = fx.copy()
    u_solve = fx.copy()

    iteration = 0
    while iteration < max_iter:
        max_u = np.max(abs(u))
        max_diff = np.max(abs(u_solve - u) + [1e-5])
        err = max_u / max_diff
        if err <= eps:
            break

        u_prev = u.copy()
        u.fill(0)

        for j in range(n + 1):
            Kj = lambda s: K(xl[j], s)

            if j != 0:
                u[j] += Kj(xl[0]) * u_prev[0]
                u[j] += Kj(xl[j]) * u_prev[j]

            inner_sum = sum(Kj(xl[k]) * u_prev[k] for k in range(1, j))
            u[j] += 2 * inner_sum

            u[j] *= h / 2

        u_solve += u
        iteration += 1

    return (xl, u_solve, h) if retstep else (xl, u_solve)



def residual(u, xl, h):
    un = np.zeros_like(u)
    for j in range(len(un)):
        Kj = lambda s: K(xl[j], s)

        un[j] += Kj(xl[0]) * u[0] + Kj(xl[j]) * u[j]

        un[j] += 2 * sum(Kj(xl[1:j]) * u[1:j])

        un[j] = h / 2 * un[j]

    return u - l * un - f(xl)


def error_show(n):
    epss = [1e-3, 1e-4, 1e-5, 1e-10, 1e-13]
    for eps in epss:
        x, u, h = solve(n, eps, retstep=True)
        ue = exact(x)
        print(f'Максимальное отличие от точного решения: {max(abs(u - ue))} \nМаксимальная невязка: {max(abs(residual(u, x, h)))}\neps = {eps} \n')



l = 1
a, b = 0, 1
n = 100
x, u6 = solve(n)
ue = exact(x)
error_show(n)
plt.figure("Решения")
plt.plot(x, u6, ".", label='Метод итерации')
plt.plot(x, ue, label='Точное')
plt.title(f"{n = }")
plt.grid()
plt.legend()
plt.show()
