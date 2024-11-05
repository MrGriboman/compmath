# https://лови5.рф/upload/uf/477/uzvy38jk0t31titcv5nft2evy4qmv20x/CHislennoe-reshenie-uravneniya-Fredgolma-2-roda-metodom-kv.pdf
import numpy as np
import matplotlib.pyplot as plt


def K(x, t):
    #return x ** 2 * np.exp(x ** 2 * t ** 4)
    return np.sin(0.6 * x * t) / (t + 1e16)


def f(x):
    #return x**3 - np.exp(x**2) + 1
    return x


def exact(x):
    #return x**3
    return 0*x



def solve(n, retstep=False):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    matrix = np.identity(n + 1)
    F = np.zeros(n + 1) + f(xl)
    A = l * np.ones(n + 1)
    A[0] = A[-1] = l / 2

    for i in range(n + 1):
        Ki = lambda s: K(xl[i], s)
        matrix[i] -= A * Ki(xl) * h

    u = np.linalg.solve(matrix, F)
    return (xl, u, h) if retstep else (xl, u)


def residual(u, xl, h):
    def res_int(x):
        Kk = lambda s: K(x, s)
        un = Kk(xl[0]) * u[0] + Kk(xl[-1]) * u[-1] + 2 * sum(Kk(xl[1:-1]) * u[1:-1])
        un = h / 2 * un
        return un

    res_i = np.array([res_int(x) for x in xl])

    return u - l * res_i - f(xl)


def error_show():
    nl = [10, 20, 50, 100, 500, 1000]
    for n in nl:
        x, u, h = solve(int(n), retstep=True)
        ue = exact(x)

        print(f'Максимальное отличие от точного решения: {max(abs(u - ue))}\nМаксимальная невязка: {max(abs(residual(u, x, h)))}\nn={n}\n')


n = 100
l = 4
a, b = 0, 1
x, u = solve(n)
ue = exact(x)
error_show()
plt.figure("Решения")
plt.plot(x, u, ".", label='Метод итерации')
plt.plot(x, ue, label='Точное')
plt.title(f"{n = }")
plt.grid()
plt.legend()
plt.show()