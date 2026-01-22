import numpy as np
import matplotlib.pyplot as plt
import time

def dft(x):
    N = len(x)
    k = np.arange(N)[:, None] #col
    n = np.arange(N)[None, :] #row
    
    x = np.reshape(x, [-1, N]) #make it row
    
    return np.sum( x * np.exp(-1j * 2 * np.pi * k * n / N ), 1)
    

def fft(x):  
    N = len(x)
    if N % 2 != 0:
        raise ValueError("N should be power of 2")
    
    if N <= 8:
        return dft(x)
    else:
        
        k = np.arange(N//2)
        f1 = np.exp(-1j * 2 * np.pi * k / N )
        f2 = -f1
            
        xEven = x[0::2]
        xOdd  = x[1::2]
        
        xHatOdd  = fft(xOdd)
        xHatEven = fft(xEven)
        
    return np.concatenate([xHatEven + xHatOdd * f1, xHatEven + xHatOdd * f2])

def ifft(xHat):
    N = len(xHat)
    # real() just to get rid of imaginary leftovers
    return np.real(np.conj(fft(np.conj(xHat))) / N)
    
    
def test():    
    x = np.linspace(0, 2 * np.pi, 16)
    y = np.sin(x)
    y2 = ifft(fft(y))

    print(np.max(np.abs(y - y2)))

def complexity(fun, N):
    #N = 14
    ns = 2**np.arange(8, N)
    y = np.random.rand(2**N)
    t = []
    
    for n in ns:
        t0 = time.perf_counter()
        for _ in range(100):
            fun(y[:n])
        t1 = time.perf_counter()
        t.append((t1 - t0)/100)
    
    
    slope = np.polyfit(np.log10(ns), np.log10(t), 1)[0]
    plt.loglog(ns, t, "o-", label=f"{fun.__name__}, slope ~ {slope:.2f}")

if __name__ == "__main__":    
    test()
    
    complexity(fft, 14)
    complexity(dft, 12)
    plt.legend()