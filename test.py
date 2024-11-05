import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Define parameters
a, b = -np.pi, np.pi   # Interval of integration
n = 50                 # Number of grid points
h = (b - a) / (n - 1)  # Step size
lambda_ = 2            # Lambda parameter in the equation

# Define the grid points
x_vals = np.linspace(a, b, n)

# Define the kernel function K(x, t)
def K(x, t):
    return np.cos(2 * x) * np.cos(t)

# Define the known function f(x)
def f(x):
    return 7 * np.sin(x)

# Set up the matrix A and vector B for the system Ay = B
A = np.eye(n)          # Start with identity matrix for diagonal (y(x) terms)
B = np.zeros(n)        # Right-hand side vector

# Populate A matrix and B vector using the trapezoidal rule
for i, x in enumerate(x_vals):
    # Calculate B vector
    B[i] = f(x)

    # Calculate each row of A
    for j, t in enumerate(x_vals):
        if i != j:
            A[i, j] = -lambda_ * K(x, t) * h
        else:
            A[i, j] += -lambda_ * K(x, t) * h / 2  # Half weight for endpoints

# Solve the system Ay = B
y_vals = solve(A, B)

# Exact solution for comparison
y_exact = np.cos(x_vals)

# Plot the numerical solution and the exact solution
plt.plot(x_vals, y_vals, label="Numerical Solution (Trapezoidal)")
plt.plot(x_vals, y_exact, label="Exact Solution $y(x) = \cos(x)$", linestyle="--")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.title("Numerical Solution vs. Exact Solution")
plt.show()

# Calculate relative error
relative_error = np.linalg.norm(y_vals - y_exact) / np.linalg.norm(y_exact)
print(f"Relative Error: {relative_error:.5f}")
