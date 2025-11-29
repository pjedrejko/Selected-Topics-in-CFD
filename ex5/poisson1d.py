import numpy as np
import matplotlib.pyplot as plt

def lap1D(x, leftBC, rightBC):
    n = len(x)
    dx = x[2] - x[1]
    L = np.zeros((n, n))

    for i in range(1, n-1):
        L[i, [i-1, i, i+1]] = [1, -2, 1] / dx**2

    if leftBC == "dirichlet":
        L[0, [0, 1]] = 0.5
    elif leftBC == "neumann":
        L[0, [0, 1]] = [-1, 1] / dx    
    elif leftBC == "periodic":
        L[0, [0, n-1]] = [-1, 1]
    else:
        raise ValueError("incorrect BC type")

    
    if rightBC == "dirichlet":
        L[n-1, [n-2, n-1]] = 0.5
    elif rightBC == "neumann":
        L[n-1, [n-2, n-1]] = [-1, 1] / dx    
    elif rightBC == "periodic":
        L[n-1, [0, n-1]] = [-1, 1]
    else:
        raise ValueError("incorrect BC type")

    return L

def neumannCompatibility(b, dx):
    #midpoint rule matching central difference
    return np.isclose( np.sum(b[1:-1]) * dx, b[-1] - b[0] )

#%%

n = 64

#domain with ghost nodes
x = np.zeros(n)
x[:-1] = np.linspace(0, np.pi, n-1)
dx = x[2]-x[1]
x -= dx/2
x[-1] = x[-2] + dx




#%%
lap = lap1D(x, "dirichlet", "dirichlet")
b = np.sin(x)
b[0]  = 0.0 #BC
b[-1] = 1.0
y = np.linalg.solve(lap, b)
plt.figure()
plt.plot(x, y)
plt.grid()
plt.title("a)")

#%%
lap = lap1D(x, "neumann", "neumann")
#analytically -> sol. determined up to const -> det(L) = 0


#better not to with iterative solvers, but solve() needs it
lap[32, :] = 0.0 #fixing the constant to whatever there is in b[32]
lap[32, 32] = 1.0

b = np.sin(x)
b[0]  = 1.0
b[-1] = 0.0
print(neumannCompatibility(b, dx))
# contradiction, so:
# lap(y[32]) = sin(x[32]) replaced with rubbish implied from other eqs. -> ugly kink

y = np.linalg.solve(lap, b)
plt.figure()
plt.plot(x, y)
plt.grid()
plt.title("b)")


#%%
lap = lap1D(x, "neumann", "neumann")

lap[32, :] = 0.0 #fixing the constant to whatever there is in b[32]
lap[32, 32] = 1.0

b = -np.sin(2*x)
b[0]  = 1.0
b[-1] = 1.0
print(neumannCompatibility(b, dx))
#now the implied condition is equivalent to lap(y[32]) = sin(x[32])

y = np.linalg.solve(lap, b)
plt.figure()
plt.plot(x, y)
plt.grid()
plt.title("c)")

