import numpy as np
import matplotlib.pyplot as plt

def lapDirichlet(x):
    n = len(x)
    dx = x[2] - x[1]
 
    L = (   np.diag(np.ones(n-1),  1) \
         -2*np.diag(np.ones(n)      ) \
          + np.diag(np.ones(n-1), -1) ) / dx**2
    
    #Dirichlet BC
    L[0, 0:2] = [1, 0]
    L[-1,-2:] = [0, 1]
    
    return L


#simplest case
def jacobi(A, y0, b, tol):
    maxIter = 20000
    checkEach = 10
    
    y1 = np.zeros_like(y0)
    
    for i in range(maxIter):
        for k in range(n):
            y1[k] = ( b[k] - ( A[k, :] @ y0 - A[k, k] * y0[k] ) ) / A[k, k]
        
        y0 = y1.copy()
        
        if(i % checkEach == 0):
            
            # /|b| to be independent of n. 
            # expected scale of |x| can be deduced from dim. analysis
            res = np.linalg.norm(A @ y1 - b) / np.linalg.norm(b)
            if( res < tol ):
               
               print(f"converged after {i} iterations")
               
               return y1
           
    print("solver didnt converge")
    return None

#records error evolution
def jacobiRecordErr(A, y0, b, tol):

    maxIter = 20000
    checkEach = 10
    
    n = len(y0)
    y1 = np.zeros_like(y0)
    
    hist = np.zeros((maxIter // checkEach, n))
    
    for i in range(maxIter):
        for k in range(n):
            y1[k] = ( b[k] - ( A[k, :] @ y0 - A[k, k] * y0[k] ) ) / A[k, k]
        
        y0 = y1.copy()
        
        if(i % checkEach == 0):
            
            res = np.linalg.norm(A @ y1 - b) / np.linalg.norm(b)
            hist[i//checkEach, :] = y1 #save current state
            
            #print(f"res: {res}")
            if( res < tol ):
               
               print(f"converged after {i} iterations")
               
               #prev. ys to errors, using last y as ~ solution
               errs = hist[:i//checkEach+1, :] - hist[i//checkEach, :]
               
               return y1, errs
           
    print("solver didnt converge")
    return None

def decomposeErr(errs, x):
    nModes = 4
    nIters = errs.shape[0]    
    
    errModes = np.zeros((nModes, nIters))

    for k in range(nModes):   
        mode = np.sin(x * 2**k) #laplacian (dirichlet BCs) eigenvec
        
        for i in range(nIters):
            
            errProjection = (errs[i, :] @ mode / (mode @ mode)) * mode
            
            errModes[k, i] = np.linalg.norm(errProjection)
                
    return errModes


def plotConv(errs, name):
    """
    err_n = lambda^n * err_0
    log(err_n) = n*log(lambda) + C
    -> show convergence as log(err) vs n
    """
    
    n = len(errs)
    lerrs = np.log10(errs)
    iters = np.arange(0, n)

    #get the slope
    logLamb = np.polyfit(iters[5:10], lerrs[5:10], 1)[0]
    lamb = 10**logLamb

    plt.plot(iters, lerrs, label = name + f", lambda ~ {lamb:.3f}")
    
    plt.xlabel("iter")
    plt.ylabel("log10(err)")
    plt.legend()

plt.close("all")

#%%

n = 64
x0 = -np.pi
x1 = np.pi
x = np.linspace(x0, x1, n)

L = lapDirichlet(x)

# RHS
b     = np.exp(-x**2) * (4*x**2 - 2)
b[0]  = np.exp(-x0**2)
b[-1] = np.exp(-x1**2)

# initial guess
y = np.random.rand((len(b)))

plt.plot(x, y)
#solving Ly = b
y, errs = jacobiRecordErr(L, y, b, 1e-8)
plt.plot(x, y)

#%%
errModes = decomposeErr(errs, x)

plt.figure()

for kMode in range(errModes.shape[0]):
    plotConv(errModes[kMode, :], f"sin({2**kMode}x)")




