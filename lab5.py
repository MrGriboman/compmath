import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.sin(0.6 * x * s)


def f(x):
    return x


def exact(x):
    return x


def get_integration_weights(size, method, h):
    if method == 'rect':
        weights = np.full(size, h) 
    elif method == 'trap':
        weights = np.ones(size) * h 
        weights[0] = weights[-1] = h / 2 
    elif method == 'simp':
        weights = np.zeros(size)
        weights[0] = weights[-1] = h / 3
        weights[1:-1:2] = 4 * h / 3
        weights[2:-2:2] = 2 * h / 3
    else:
        raise ValueError("Method must be 'rect', 'trap', or 'simp'.")

    return weights


a, b = 0, 3 * np.pi
n = 101
xs, h = np.linspace(a, b, n + 1, retstep=True)
fs = f(xs)

for meth in ['rect', 'trap', 'simp']:
    w = get_integration_weights(n + 1, meth, h)

    u = np.zeros(n + 1)
    for i in range(n + 1):
        u[i] = (fs[i] + sum(w * K(xs[i], xs) * u)) / (1 - w[i] * K(xs[i], xs[i]))


    #print(f'Максимальная ошибка ({meth}): {max(abs(u - exact(xs)))}')

    plt.figure("Решение")
    plt.plot(xs, u, label=f'Квадратуры ({meth})')
    #plt.plot(xs, exact(xs), "--", label='Точное')
    plt.legend()
    plt.show()