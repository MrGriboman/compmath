import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from TDMA import TDMA


def exact_sol(x, y, t):
    return t * np.exp(x + y)


def f(t, x, y):
    return (1 - 2 * t) * np.exp(x + y)


Lx, Ly = 1, 1  
Nx, Ny = 50, 50 
T = 1 
Nt = 200 
alpha = 1

x, dx = np.linspace(0, Lx, Nx, retstep=True)
y, dy = np.linspace(0, Ly, Ny, retstep=True)
t, dt = np.linspace(0, T, Nt+1, retstep=True)

gamma_x = alpha * dt / dx**2
gamma_y = alpha * dt / dy**2

u = np.zeros((Nt+1, Nx, Ny))


for n in range(Nt):
    u_half = np.zeros((Nx, Ny))
    for j in range(1, Ny-1):
        a = -0.5 * gamma_x * np.ones(Nx-1)
        b = (1 + gamma_x) * np.ones(Nx)
        c = -0.5 * gamma_x * np.ones(Nx-1)
        d = u[n, :, j] + 0.5 * dt * f(t[n] + 0.5 * dt, x, y[j])
        
        d[0] = t[n] * np.exp(0 + y[j])
        d[-1] = t[n] * np.exp(1 + y[j])
        u_half[:, j] = TDMA(a, b, c, d)
    
    for i in range(1, Nx-1):
        a = -0.5 * gamma_y * np.ones(Ny-1)
        b = (1 + gamma_y) * np.ones(Ny)
        c = -0.5 * gamma_y * np.ones(Ny-1)
        d = u_half[i, :] + 0.5 * dt * f(t[n] + dt, x[i], y)
        
        d[0] += 0.5 * gamma_y * u_half[i, 1]
        d[-1] += 0.5 * gamma_y * u_half[i, -2]
        u[n+1, i, :] = TDMA(a, b, c, d)
    
    u[n+1, 0, :] = 0
    u[n+1, -1, :] = 0




X, Y = np.meshgrid(x, y, indexing="ij")
Z = u[-5]

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Hot")])

fig.update_layout(
    title="Temperature Distribution at Final Time",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="Temperature"
    ),
    coloraxis_colorbar=dict(title="Temperature")
)

fig.show()