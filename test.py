import numpy as np
import matplotlib.pyplot as plt

# Define parameters
Lx, Ly = 1, 1  # Lengths in x and y directions
Nx, Ny = 50, 50  # Number of grid points in x and y directions
T = 1  # Total time
Nt = 200  # Number of time steps
alpha = 1  # Diffusion coefficient in the PDE

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / Nt

# Stability parameters
gamma_x = alpha * dt / dx**2
gamma_y = alpha * dt / dy**2

# Create the grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
t = np.linspace(0, T, Nt+1)

# Initialize the solution grid
u = np.zeros((Nt+1, Nx, Ny))

# Define the source function
def f(t, x, y):
    return (1 - 2 * t) * np.exp(x + y)

# Thomas algorithm for solving tridiagonal systems
def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_star = np.zeros(n-1)
    d_star = np.zeros(n)
    
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * c_star[i-1]
        c_star[i] = c[i] / denom
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom
    
    d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2])
    
    x = np.zeros(n)
    x[-1] = d_star[-1]
    for i in reversed(range(n-1)):
        x[i] = d_star[i] - c_star[i] * x[i+1]
    
    return x

# Time-stepping loop
for n in range(Nt):
    current_time = t[n]
    
    # Step 1: Half-step in the x-direction (implicit in x, explicit in y)
    u_half = np.zeros((Nx, Ny))
    for j in range(1, Ny-1):
        # Construct the tridiagonal system
        a = -0.5 * gamma_x * np.ones(Nx-1)
        b = (1 + gamma_x) * np.ones(Nx)
        c = -0.5 * gamma_x * np.ones(Nx-1)
        d = u[n, :, j] + 0.5 * dt * f(current_time + 0.5 * dt, x, y[j])
        
        # Apply Dirichlet boundary conditions in x
        d[0] = current_time * np.exp(y[j])           # Left boundary at x = 0
        d[-1] = current_time * np.exp(1 + y[j])      # Right boundary at x = 1
        u_half[:, j] = thomas_algorithm(a, b, c, d)
    
    # Step 2: Full step in the y-direction (implicit in y, explicit in x)
    for i in range(1, Nx-1):
        # Construct the tridiagonal system
        a = -0.5 * gamma_y * np.ones(Ny-1)
        b = (1 + gamma_y) * np.ones(Ny)
        c = -0.5 * gamma_y * np.ones(Ny-1)
        d = u_half[i, :] + 0.5 * dt * f(current_time + dt, x[i], y)
        
        # Apply Dirichlet boundary conditions in y
        d[0] = current_time * np.exp(x[i])           # Bottom boundary at y = 0
        d[-1] = current_time * np.exp(x[i] + 1)      # Top boundary at y = 1
        u[n+1, i, :] = thomas_algorithm(a, b, c, d)
    
    # Update boundary conditions for the solution grid
    u[n+1, 0, :] = current_time * np.exp(y)          # Left boundary in x
    u[n+1, -1, :] = current_time * np.exp(1 + y)     # Right boundary in x
    u[n+1, :, 0] = current_time * np.exp(x)          # Bottom boundary in y
    u[n+1, :, -1] = current_time * np.exp(x + 1)     # Top boundary in y

# Example: Visualize the final temperature distribution
X, Y = np.meshgrid(x, y, indexing="ij")
plt.contourf(X, Y, u[-1].T, 20, cmap="hot")
plt.colorbar(label="Temperature")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Temperature distribution at final time")
plt.show()