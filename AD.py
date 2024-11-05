import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def exact(x, y, t):
    return t * np.exp(x + y)


def f(x, y, t):
    return (1 - 2 * t) * np.exp(x + y)


a, b = 0, 1
n = 10
t_steps = 1000
dx = (b - a) / (n - 1)
dt = (b - a) / t_steps

x_vals = np.linspace(a, b, n)
y_vals = np.linspace(a, b, n)
t_vals = np.linspace(a, b, t_steps)

A_x = np.zeros((n, n))
A_y = np.zeros((n, n))

np.fill_diagonal(A_x, 2 / dx**2)
np.fill_diagonal(A_y, 2 / dx**2)

A_x[np.arange(1, n), np.arange(n - 1)] = -1 / dx**2
A_y[np.arange(1, n), np.arange(n - 1)] = -1 / dx**2

A_x[np.arange(n - 1), np.arange(1, n)] = -1 / dx**2
A_y[np.arange(n - 1), np.arange(1, n)] = -1 / dx**2


u = np.zeros((t_steps, n, n))
ex = np.zeros((t_steps, n, n))

X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
u[0] = exact(X, Y, 0)

T = t_vals[:, None, None]
ex = exact(X, Y, T)


I = np.identity(n)
matrix_x = (1 / dt * I + 0.5 * A_x)
matrix_y = (1 / dt * I + 0.5 * A_y)

matrix_x[0, :2] = [1, 0]
matrix_x[-1, -2:] = [0, 1]
matrix_y[0, :2] = [1, 0]
matrix_y[-1, -2:] = [0, 1]

matrix_x_rhs = (1 / dt * I - 0.5 * A_x)
matrix_y_rhs = (1 / dt * I - 0.5 * A_y)

temp_phi = np.zeros((1, n, n))
rhs_x = np.zeros((n, n))
rhs_y = np.zeros((n, n))
f_values = np.zeros((n, n))

for t_idx in range(1, t_steps):
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")
    f_values = f(X, Y, (t_vals[t_idx - 1] + t_vals[t_idx]) / 2)
    rhs_x = matrix_x_rhs @ u[t_idx - 1].T
    rhs_x += f_values / 2

    # граничные условия для правой части (1 прогонка)
    rhs_x[:, 0] = exact(x_vals, y_vals[0], (t_vals[t_idx - 1] + t_vals[t_idx]) / 2)
    rhs_x[:, -1] = exact(x_vals, y_vals[-1], (t_vals[t_idx - 1] + t_vals[t_idx]) / 2)
    temp_phi[0] = np.array([solve(matrix_x, rhs_x[i]) for i in range(n)])

    rhs_y = matrix_y_rhs @ temp_phi[0].T
    rhs_y += f_values / 2

    # граничные условия для правой части (2 прогонка)
    rhs_y[0, :] = exact(x_vals[0], y_vals, t_vals[t_idx])
    rhs_y[-1, :] = exact(x_vals[-1], y_vals, t_vals[t_idx])
    u[t_idx] = np.array([solve(matrix_y, rhs_y[:, i]) for i in range(n)]).T

    # граничные значения
    u[t_idx, :, 0] = exact(x_vals, y_vals[0], t_vals[t_idx])
    u[t_idx, :, -1] = exact(x_vals, y_vals[-1], t_vals[t_idx]) 
    u[t_idx, 0, :] = exact(x_vals[0], y_vals, t_vals[t_idx])
    u[t_idx, -1, :] = exact(x_vals[-1], y_vals, t_vals[t_idx])


errors = np.max(np.abs(ex - u), axis=(1, 2))
print("Maximum Error: ", max(errors))

X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals)

fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], 
                    subplot_titles=("Численное", "Точное"))


fig.add_trace(go.Surface(z=u[-2], x=X_mesh, y=Y_mesh, colorscale='Jet', showscale=False), row=1, col=1)

fig.add_trace(go.Surface(z=ex[-1], x=X_mesh, y=Y_mesh, colorscale='Jet', showscale=False), row=1, col=2)

fig.update_layout(
    title="Метод переменных направлений",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
