import numpy as np
import matplotlib.pyplot as plt

N = 512 # a lot cause we do not dynamically refine here
xi = np.linspace(0, 2 * np.pi, N)
x = np.array([np.sin(xi), np.cos(xi)]).T
u = np.zeros_like(x)
gamma = -np.sin(xi)
deltaSq = 0.1

def biotSavart(x0):
        rSq = np.sum( (x0 - x)**2, 1) + deltaSq
        
        u0 = np.trapezoid( gamma[:, None] * (x0 - x) / (2 * np.pi * rSq[:, None]), xi, axis=0)
        
        return np.array([-u0[1], u0[0]])

def computeVelocities(u):
    for i in range(N):
        u[i, :] = biotSavart(x[i, :])
        
def timeStep(x):
    dt = 0.8 * np.sqrt(deltaSq) / np.max(np.linalg.norm(u, axis=1))
    x += u * dt
    
def plotSheet():
    plt.plot(x[:,0], x[:,1], "k-")
    plt.scatter(x[:,0], x[:,1], s=20, c=-gamma, cmap='RdBu', zorder=10)
    
    
    
for i in range(30):
    plt.clf()
    computeVelocities(u)
    timeStep(x)
    
    plotSheet()  
    plt.axis('scaled')
    plt.ylim(-1, 5)
    plt.xlim(-2, 2)  

    
    plt.pause(0.01)
    