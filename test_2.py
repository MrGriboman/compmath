import numpy as np
from scipy import integrate
import sympy

def f_0(x):
    return x

def A_int(s, k):
    return phi(s, k - 1)

def phi(f, k):
    for _ in range(k - 1):
        f = lambda x, f=f: integrate.quad(f, 0, x)
    return f
    
    
a = phi(f_0, 2)
print(a)
