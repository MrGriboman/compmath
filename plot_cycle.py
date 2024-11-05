import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp


def exact_sol(x, y, t):
    return t * np.exp(x + y)


def f(x, y, t):
    return (1 - 2 * t) * np.exp(x + y)


def step_mat(l, vals):
    identity_matrix = np.eye(len(l))
    rhs_vector = np.dot((identity_matrix - (dt * l) / 2), vals)
    lhs_matrix = identity_matrix + (dt * l) / 2
    result = np.linalg.solve(lhs_matrix, rhs_vector)
    return result


def l_oper(size):
    mat = np.zeros((size + 1, size + 1))
    mat = mat - np.diag(np.zeros(size + 1) + 2)
    
    for i in range(1, size):
        mat[i, i - 1] = 1
        mat[i, i + 1] = 1

    mat[0, 0], mat[0, 1], mat[0, 2] = 1, -2, 1
    mat[-1, -3], mat[-1, -2], mat[-1, -1] = 1, -2, 1
    return mat


t_start, t_end = 0.001, 0.1
x_start, x_end = 0, 1
y_start, y_end = 0, 1
n_t, n_x, n_y = 30, 10, 10
t_vals, dt = np.linspace(t_start, t_end, n_t + 1, retstep=True)
x_vals, dx = np.linspace(x_start, x_end, n_x + 1, retstep=True)
y_vals, dy = np.linspace(y_start, y_end, n_y + 1, retstep=True)

u = np.zeros((n_t + 1, n_y + 1, n_x + 1))
u_exact = np.zeros((n_t + 1, n_y + 1, n_x + 1))
X, Y = np.meshgrid(x_vals, y_vals)

Lx = -l_oper(n_x) / dx**2
Ly = -l_oper(n_y) / dy**2

u[0] = exact_sol(X, Y, t_start)
u[-1] = exact_sol(X, Y, t_end)
h_step = np.zeros((n_y + 1, n_x + 1))
for t in range(1, n_t + 1, 2):
    h_step = np.array([step_mat(Lx, u[t - 1, idx_y]) for idx_y in range(n_y + 1)])
    np.copyto(u[t], h_step)

    temp_step = np.empty_like(h_step)
    for idx_x in range(n_x + 1):
        temp_step[:, idx_x] = step_mat(Ly, u[t][:, idx_x])

    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    u[t] = temp_step + dt * f(x_mesh, y_mesh, t_vals[t])

    for idx_x in range(n_x + 1):
        temp_step[:, idx_x] = step_mat(Ly, u[t][:, idx_x] + dt * f(x_vals[idx_x], y_vals, t_vals[t]))
    u[t + 1] = np.copy(temp_step)

    h_step = np.array([step_mat(Lx, u[t + 1, idx_y]) for idx_y in range(n_y + 1)])
    np.copyto(u[t + 1], h_step)

    u[t, :, 0] = exact_sol(x_vals, y_vals[0], t_vals[t])
    u[t, :, -1] = exact_sol(x_vals, y_vals[-1], t_vals[t]) 
    u[t, 0, :] = exact_sol(x_vals[0], y_vals, t_vals[t])
    u[t, -1, :] = exact_sol(x_vals[-1], y_vals, t_vals[t])


T_vals = t_vals[:, np.newaxis, np.newaxis]
u_exact = T_vals * np.exp(X + Y)

error = np.zeros(n_t + 1)
for t in range(n_t + 1):
    error[t] = np.max(abs(u[t] - u_exact[t]))


def plot_surface(z):
    return go.Surface(z=z, x=x_vals, y=y_vals, colorscale='Plasma', showscale=False)


fig = sp.make_subplots(rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                       subplot_titles=["Численное", "Точное"])

for t in range(10):
    fig.add_trace(plot_surface(u[t]), row=1, col=1)
    fig.add_trace(plot_surface(u_exact[t]), row=1, col=2)

fig.update_layout(title="Уравнение теплопроводности. Метод 2-циклического покомпонетного расщепления",
                  scene=dict(zaxis=dict(range=[np.min([u, u_exact]) - 2 * dt, np.max([u, u_exact]) + 2 * dt])),
                  height=700)

fig.show()



error_fig = go.Figure()
error_fig.add_trace(go.Scatter(x=t_vals, y=error, mode='lines+markers', name="Error"))
error_fig.update_layout(title="Максимальная ошибка",
                        xaxis_title="Время (t)",
                        yaxis_title="Ошибка (max |u - exact|)",
                        height=500)

error_fig.show()