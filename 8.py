import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.integrate import quad


def K(x, s):
    return x ** 2 * np.exp(x ** 2 * s ** 4)    
    return np.sin(0.6*x*s) / (s + 1e16)

# alpha
def Kx(i):
    return lambda x: x ** (2 + 2 * i)    
    return lambda x: (0.6*x) ** (2*i + 1)

# beta
def Ks(i):
    return lambda s: s ** (4 * i) / factorial(i)    
    return lambda s: ((-1)**(i) * s**(2*i)) / factorial(2*i + 1)


def f(x):
    return x ** 3 - np.exp(x ** 2) + 1    
    return x

def exact(x):
    return x ** 3    
    return 0 * x


def solve(n=10):
    F = np.zeros(n)
    A = np.zeros((n,n))

    for j in range(n):
        F[j] = quad(lambda s: f(s) * Ks(j)(s), a, b)[0]
        for i in range(n):
            A[j, i] = quad(lambda s: Kx(i)(s) * Ks(j)(s), a, b)[0]

    mat = np.eye(n) - l * A
    C = np.linalg.solve(mat, F)

    return lambda x: f(x) + l * sum(C[i] * Kx(i)(x) for i in range(n))

def residual(u, xl, h):
    def res_int(x):
        Kk = lambda s: K(x, s)
        un = Kk(xl[0]) * u[0] + Kk(xl[-1]) * u[-1] + 2 * sum(Kk(xl[1:-1]) * u[1:-1])
        un = h / 2 * un
        return un

    res_i = np.array([res_int(x) for x in xl])

    return u - l * res_i - f(xl)


def error_show():
    xn = 1000
    xl, h = np.linspace(a, b, xn + 1, retstep=True)

    ue = np.zeros(xn + 1) + exact(xl)

    nl = [1, 2, 3, 5, 10]
    for n in nl:
        u = solve(n)
        u_v = u(xl)

        nevf = lambda x: u(x) - l * quad(lambda s: K(x, s) * u(s), a, b)[0] - f(x)
        nev = np.zeros_like(xl)
        for i, x in enumerate(xl):
            nev[i] = nevf(x)

        print(f'n = {n}\nМаксимальное отличие от точного решения: {max(abs(u_v - ue))}\nМаксимальная невязка: {max(abs(residual(u_v, xl, h)))}\n')


n = 100
l = 4
a, b = 0, 1
x = np.linspace(a, b, n + 1)
u_f = solve()
u = u_f(x)
ue = exact(x)
error_show()
plt.figure("Решения")
plt.plot(x, u, ".", label='Метод вырожденных ядер')
plt.plot(x, ue, label='Точное')
plt.title(f"{n = }")
plt.grid()
plt.legend()
plt.show()