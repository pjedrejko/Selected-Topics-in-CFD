import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# avoid storing the whole matrix for speed (data locality)
# store just the diagonals
class triDiagMatrix:
    def __init__(self, downDiag, diag, upDiag):
        assert(len(downDiag) == len(upDiag) == len(diag)-1)

        self.downDiag = downDiag
        self.diag = diag
        self.upDiag = upDiag


def jacobi(A: triDiagMatrix, u, b, tol):
    maxIter = 1_000_000
    
    u2  = np.zeros_like(u)
    res = np.zeros_like(u)
    
    for i in range(maxIter):
        # vectorized operations for speed
        u2       = b.copy()
        u2[:-1] -= A.upDiag   * u[1:]
        u2[1:]  -= A.downDiag * u[:-1]
        u2      /= A.diag
        
        u = u2.copy()
        
        if i % 50 == 0:
            res = A.diag * u - b
            res[:-1] += A.upDiag   * u[1:]
            res[1:]  += A.downDiag * u[:-1]
            
            
            if(np.linalg.norm(res) / np.linalg.norm(b) < tol):
                print(f"{i},  {np.linalg.norm(res)}")
                return u, i

#%% solve the same probelm for various numbers of grid points
Ns = 2**np.arange(3, 10)
nIters = np.zeros_like(Ns) #how many iterations Jacobi needs

for i in range(len(Ns)):
    N = Ns[i]
 
    # sparse operator
    diag = -2*np.ones(N)
    diag[[0, -1]] = 1 #dirichlet
    A = triDiagMatrix(np.ones(N-1), diag, np.ones(N-1))
    
    # domain
    x0 = -np.pi
    x1 = np.pi
    x = np.linspace(-np.pi, np.pi, N)
    
    # RHS
    b     = np.exp(-x**2) * (4*x**2 - 2)
    b[0]  = np.exp(-x0**2)
    b[-1] = np.exp(-x1**2)
    
    #solve with initial guess = 0
    u, nIters[i] = jacobi(A, np.zeros(N), b, 1e-6)

#%% get nIters(N), should be quadratish

plt.plot(np.log10(Ns), np.log10(nIters), "o-")
plt.axis("equal")
plt.xlabel("log10 N")
plt.ylabel("log10 nIters")
slope = np.polyfit(np.log10(Ns), np.log10(nIters), 1)[0] #fit linear function
plt.title(f"slope = {slope}")

