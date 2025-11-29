import numpy as np
import matplotlib.pyplot as plt

def idx(nCols, i, j):
    return nCols * i + j

def lap2D(x, y):
    nx = len(x)
    ny = len(y)
    N = nx * ny
    
    dx = x[2] - x[1]
    dy = y[2] - y[1]
    
    L = np.zeros((N, N))

    for i in range(1, ny-1):
        for j in range(1, nx-1):
            L[idx(nx,i,j), [idx(nx,i-1,j), idx(nx,i,j), idx(nx,i+1,j)]] += [1, -2, 1] / dx**2
            L[idx(nx,i,j), [idx(nx,i,j-1), idx(nx,i,j), idx(nx,i,j+1)]] += [1, -2, 1] / dy**2
            
    return L

#more efficient than building operator explicitly
def computeLap2D(U, dx, dy):
    return                     \
     1 * U[ :-2, 1:-1] / dy**2 \
    -2 * U[1:-1, 1:-1] / dy**2 \
    +1 * U[2:  , 1:-1] / dy**2 \
    +1 * U[1:-1,  :-2] / dx**2 \
    -2 * U[1:-1, 1:-1] / dx**2 \
    +1 * U[1:-1, 2:  ] / dx**2


def setDirichlet(uIn, uOut, dx, val):
    # (uOut + uIn)/2 = val
    uOut[:] = 2.0 * val - uIn #[:] to modify view's content not reassign it

def setNeumann(uIn, uOut, dx, val):
    # derivative along domain outward dir.
    # (uOut - uIn)/dx = val
    uOut[:] = dx * val + uIn


plt.close("all")
n = 32
L = np.pi
dx = L / (n-1)
alpha = 1.0

Co = 0.5
dt = 0.25 * dx**2 / alpha * Co #2d Courant condition

x = np.arange(-dx/2, L+dx/2, dx)
X, Y = np.meshgrid(x, x)

U = np.cos(X) * np.cos(Y)

lap = lap2D(x, x)

plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, U, cmap = "coolwarm")
plt.axis("equal")


T = 2

for i in range(int(T/dt)):
    
    #U.ravel()[:] += alpha * dt * lap @ U.ravel()
    U[1:-1, 1:-1] += alpha * dt * computeLap2D(U, dx, dx)     
       
    setNeumann(U[:,1],  U[:,0],  dx, 0.0) #west
    setNeumann(U[:,-2], U[:,-1], dx, 0.0) #east
    setNeumann(U[1,:],  U[0,:],  dx, 0.0) #south
    setNeumann(U[-2,:], U[-1,:], dx, 0.0) #north
    
    if( i % 10 == 0 ):
        ax.clear()    
        ax.plot_surface(X, Y, U, cmap = "coolwarm", vmin=-1, vmax=1)
        ax.set_zlim(-1, 1)
        plt.pause(0.1)  

