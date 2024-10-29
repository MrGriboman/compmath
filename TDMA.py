import numpy as np

# Thomas algorithm for solving tridiagonal systems
def TDMA(a, b, c, d):
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