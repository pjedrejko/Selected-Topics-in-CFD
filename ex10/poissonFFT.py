import numpy as np
import matplotlib.pyplot as plt
from CooleyTukey import fft, ifft

N = 64

L = 2 * np.pi
x = np.linspace(0, L, N+1)[:N] #to exclude last node
f = np.sin(x) + np.cos(3 * x) - np.sin(4 * x)
ySolution = -np.sin(x) - np.cos(3*x)/9 + np.sin(4*x)/16

k = np.arange(N)
#map k from [0, N) to (N/2, N/2] (aliasing)
k[k>N//2] -= N

# d( exp(-i * 2*pi * k * x / L) ) / dx = -i * 2*pi * k / L = - i * kappa
kappa = 2 * np.pi * k / L

#y'' = f
#F{y} * (i * kappa)^2 = F{f}
yHat = np.zeros_like(x, dtype=np.complex128)

yHat[1:] = fft(f)[1:] / (1j * kappa[1:])**2 #[1:] - leave 0th mode = 0 (fix mean value)
y = ifft(yHat) 

plt.plot(x, ySolution)
plt.plot(x, y, "yo")

