import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from TDMA import TDMA


def exact_sol(x, y, t):
    return t * np.exp(x + y)


def f(t, x, y):
    return (1 - 2 * t) * np.exp(x + y)


#def f(t, x, y):
#    return np.exp(t) * (x**2 - 1) * np.cos(y)


# Define parameters
Lx, Ly = 1, 1  # Lengths in x and y directions
Nx, Ny = 50, 50  # Number of grid points in x and y directions
T = 1  # Total time
Nt = 200  # Number of time steps
alpha = 1  # Diffusion coefficient in the PDE

# Create the grid
x, dx = np.linspace(0, Lx, Nx, retstep=True)
y, dy = np.linspace(0, Ly, Ny, retstep=True)
t, dt = np.linspace(0, T, Nt+1, retstep=True)

# Stability parameters
gamma_x = alpha * dt / dx**2
gamma_y = alpha * dt / dy**2

# Initialize the solution grid
u = np.zeros((Nt+1, Nx, Ny))


# Time-stepping loop
for n in range(Nt):
    # Step 1: Half-step in the x-direction (implicit in x, explicit in y)
    u_half = np.zeros((Nx, Ny))
    for j in range(1, Ny-1):
        # Construct the tridiagonal system
        a = -0.5 * gamma_x * np.ones(Nx-1)
        b = (1 + gamma_x) * np.ones(Nx)
        c = -0.5 * gamma_x * np.ones(Nx-1)
        d = u[n, :, j] + 0.5 * dt * f(t[n] + 0.5 * dt, x, y[j])
        
        # Apply Dirichlet boundary conditions in x
        d[0] = t[n] * np.exp(0 + y[j])
        d[-1] = t[n] * np.exp(1 + y[j])
        u_half[:, j] = TDMA(a, b, c, d)
    
    # Step 2: Full step in the y-direction (implicit in y, explicit in x)
    for i in range(1, Nx-1):
        # Construct the tridiagonal system
        a = -0.5 * gamma_y * np.ones(Ny-1)
        b = (1 + gamma_y) * np.ones(Ny)
        c = -0.5 * gamma_y * np.ones(Ny-1)
        d = u_half[i, :] + 0.5 * dt * f(t[n] + dt, x[i], y)
        
        # Apply Neumann boundary conditions in y
        d[0] += 0.5 * gamma_y * u_half[i, 1]
        d[-1] += 0.5 * gamma_y * u_half[i, -2]
        u[n+1, i, :] = TDMA(a, b, c, d)
    
    # Boundary conditions for the solution grid
    u[n+1, 0, :] = 0
    u[n+1, -1, :] = 0




# Create a meshgrid for x and y
X, Y = np.meshgrid(x, y, indexing="ij")
Z = u[-5]  # Use the temperature distribution at the final time step

# Create a 3D surface plot with Plotly
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Hot")])

# Update layout for readability
fig.update_layout(
    title="Temperature Distribution at Final Time",
    scene=dict(
        xaxis_title="x",
        yaxis_title="y",
        zaxis_title="Temperature"
    ),
    coloraxis_colorbar=dict(title="Temperature")
)

# Show plot
fig.show()