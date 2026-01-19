import numpy as np
import matplotlib.pyplot as plt
plt.close("all")


def central(f, u, dx, dt):
    return f -dt * u * (np.roll(f, -1) - np.roll(f, 1)) / (2*dx)

def upWind1(f, u, dx):

    uP = np.maximum(u, np.zeros_like(u))
    uM = np.minimum(u, np.zeros_like(u))
    
    fM = (f - np.roll(f, 1) ) / dx
    fP = (np.roll(f, -1) - f) / dx
    
    return -uP * fM + uM * fP

def upWind2(f, u, dx):

    uP = np.maximum(u, np.zeros_like(u))
    uM = np.minimum(u, np.zeros_like(u))
    
    fM = ( 3*f - 4*np.roll(f, 1) + np.roll(f, 2) ) / (2*dx)
    fP = (-np.roll(f, -2) + 4*np.roll(f, -1) - 3*f) / (2*dx)
    
    return -uP * fM + uM * fP

def F(f1, f2, u, dx, dt):
    return ( (u + np.abs(u)) * f1 + (u - np.abs(u)) * f2 ) * dt / (2*dx)

def upStream(f, u, dx, dt):
    return f - (\
                 F(np.roll(f, 0), np.roll(f, -1), np.roll(u, -1), dx, dt) \
                -F(np.roll(f, 1), np.roll(f,  0), np.roll(u,  0), dx, dt) \
                ) 

def smolark(f, u, dx, dt):
    fStar = upStream(f, u, dx, dt)
    
    uTilde = (np.abs(u) * dx - dt * u**2) * (np.roll(f, -1) - f) / (np.roll(f, -1) + f +1e-15) / dx
    
    
    return upStream(fStar, uTilde, dx, dt)
    

n = 64
x = np.linspace(0.0, 5.0, n)
dx = x[1] - x[0]

f0 = np.zeros(n)

f0 = 4 * (x-1)**2 * (2-x)**2
f0[(x<1) | (x>2)] = 0.0
f0Max = np.max(f0)
u = np.ones(n)
dt = dx / u * 0.4

for scheme in [central, upStream, smolark]:
    plt.figure()
    
    f = f0.copy()
    
    for i in range(60):
        f = scheme(f, u, dx, dt)
        
        plt.clf()
        plt.plot(x, f)
        plt.plot(x, f0, "k")
        plt.plot(x, np.ones_like(x) * f0Max, "k--")
        plt.xlim([0, 5])
        plt.ylim([-0.2, 0.3])
        
        
        fErr  = 100 * (np.sum(f0) - np.sum(f) ) / np.sum(f0)
        f2Err = 100 * (np.sum(f0**2) - np.sum(f**2) ) / np.sum(f0**2)
        plt.title(scheme.__name__ + f"\nerror($f$) = {fErr:.2e}%, error($f^2$) = {f2Err:.2e}%")

        plt.pause(0.01)