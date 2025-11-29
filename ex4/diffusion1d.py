import numpy as np
import matplotlib.pyplot as plt

def lap1D(x):
    n = len(x)
    dx = x[2] - x[1]
    L = np.zeros((n, n))

    for i in range(1, n-1):
        L[i, [i-1, i, i+1]] = [1, -2, 1] / dx**2

    return L


def setDirichlet(uIn, uOut, dx, val):
    # (uOut + uIn)/2 = val
    uOut[:] = 2.0 * val - uIn #[:] to modify view's content not reassign it

def setNeumann(uIn, uOut, dx, val):
    # derivative along domain outward dir.
    # (uOut - uIn)/dx = val
    uOut[:] = dx * val + uIn

#%%
def diffusion(Co, bcFuncL, bcValL, bcFuncR, bcValR):
    plt.figure()
    plt.title(f"Co = {Co}")
    plt.grid()
    
    
    n = 64
    L = np.pi
    dx = L / (n-1)
    alpha = 1.0
    
    # Courant condition (eigvals of update matrix < 1)
    dt = 0.5 * dx**2 / alpha * Co
    
    x = np.arange(-dx/2, L+dx/2, dx)
    u = np.cos(x) + 1
       
    L = lap1D(x)
                
    
    t = 0.0
    T = 1
    tSnap = np.linspace(0, T, 4)
    
    while t < T:
        u = u + alpha * dt * L @ u
        #setNeumann(u[1:2],   u[:1 ], dx, 0.0)
        #setNeumann(u[-2:-1], u[-1:], dx, 0.0)
        bcFuncL(u[1:2],   u[0:1],   dx, bcValL)
        bcFuncR(u[-2:-1], u[-1:], dx, bcValR)
            
        if any( (t <= tSnap) & (t+dt >= tSnap) ):
            uTotal = np.sum(u[1:-1])*dx #excluding ghosts
            plt.plot(x, u, label = f"uTotal({t:.2f}) = {uTotal}")
         
        t += dt
        
    plt.legend()
     
#%%
plt.close("all")

#stable
diffusion(0.5, setNeumann, 0, setNeumann, 0)

diffusion(0.5, setDirichlet, 2, setDirichlet, 0)

#%%

#blow up (but still conservative)
diffusion(1.008, setNeumann, 0, setNeumann, 0)

diffusion(1.01, setNeumann, 0, setNeumann, 0)


