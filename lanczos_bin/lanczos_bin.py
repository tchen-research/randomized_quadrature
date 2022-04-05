import numpy as np
import scipy as sp


def exact_lanczos(A,q0,k,reorth=True):
    """
    run Lanczos with reorthogonalization
    
    Input
    -----
    A : entries of diagonal matrix A
    q0 : starting vector
    k : number of iterations
    B : entries of diagonal weights for orthogonalization
    """
    
    n = len(q0)
    
    Q = np.zeros((n,k),dtype=A.dtype)
    a = np.zeros(k,dtype=A.dtype)
    b = np.zeros(k-1,dtype=A.dtype)
    
    Q[:,0] = q0 / np.sqrt(q0.T@q0)
    
    for i in range(1,k+1):
        # expand Krylov space
        qi = A@Q[:,i-1] - b[i-2]*Q[:,i-2] if i>1 else A@Q[:,i-1]
        
        a[i-1] = qi.T@Q[:,i-1]
        qi -= a[i-1]*Q[:,i-1]
        
        if reorth:
            qi -= Q@(Q.T@qi) # regular GS
            #for j in range(i-1): # modified GS (a bit too slow)
            #    qi -= (qi.T@Q[:,j])*Q[:,j]
            
        if i < k:
            b[i-1] = np.sqrt(qi.T@qi)
            Q[:,i] = qi / b[i-1]
                
    return Q,(a,b)

"""
def Psi(x,nodes,weights):
    if np.isscalar(x):
        return np.sum((nodes<=x)*weights)
    else:
        return np.sum((nodes<=x[:,None])*weights,axis=1)
    
def Psi_l(x,nodes,weights):
    if np.isscalar(x):
        return np.sum((nodes[1:]<=x)*weights[:-1])
    else:
        return np.sum((nodes[1:]<=x[:,None])*weights[:-1],axis=1)
    
def Psi_u(x,nodes,weights):
    if np.isscalar(x):
        return weights[0]+np.sum((nodes[:-1]<=x)*weights[1:])
    else:
        return weights[0]+np.sum((nodes[:-1]<=x[:,None])*weights[1:],axis=1)
"""

def Riemann_Stieltjes(f,x,Fx,right_sc=True):
    """
    compute Riemann-Stieltjes integral of f against distribution (x,Fx)
    
    """
    if right_sc:
        dFx = np.diff(Fx,prepend=Fx[0])
    else:
        dFx = np.diff(Fx,append=Fx[-1])
        
    return np.sum(f(x)*dFx)


def density_to_distribution(x,nodes,weights):
    """
    returns distribution for given density
    
    Parameters
    ----------
    x : scalar or ndarray 
        location to evaluate distribution
    nodes : (n,) ndarray
            support of density
    weights : (n,) ndarray 
              weights of density
    
    Returns
    -------
    Fx : ndarray 
         right continuous density evaluated at x
    
    """
    if np.isscalar(x):
        return np.sum((nodes<=x)*weights)
    else:
        if np.all(x==nodes):
            return np.cumsum(weights)
        
        return np.sum((nodes<=x[:,None])*weights,axis=1)
    
def average_density(supports,weights):
    """
    returns average density 
    
    Parameters
    ----------
    supports : (n_samples,n) ndarray
    weights : (n_samples,n) ndarray
    
    Returns
    -------
    weights : (n,) ndarray 
    """
    
    order = np.argsort(supports.flatten())
    
    return supports.flatten()[order],weights.flatten()[order]/weights.shape[0]
    
def gq_nodes_weights(a,b):
    """
    get quadrature nodes and weights from jacobi matrix
    
    Parameters
    ----------
    a : (k,) ndarray
    b : (k-1,) ndarray
    
    Returns
    -------
    nodes : (k+2,) ndarray
    weights : (k+2,) ndarray
    """
    theta,S = sp.linalg.eigh_tridiagonal(a,b,tol=1e-30)

    return np.hstack([np.nextafter(theta[0],-np.inf),theta,np.nextafter(theta[-1],np.inf)]),\
           np.hstack([0,S[0]**2,0])
    
def gq_lower_density(nodes,weights,lb,ub):
    """
    get denisty corresponding to lower bounds for Gaussian quadrature density
    
    Parameters
    ----------
    nodes : (k,) ndarray
    weights : (k,) ndarray
    
    Returns
    -------
    nodes : (k+2,) ndarray
    weights : (k+2,) ndarray
    """
    
    return np.hstack([nodes[1:-1],ub,np.nextafter(ub,np.inf)]),\
           np.hstack([0,weights[1:-1],0])

def gq_upper_density(nodes,weights,lb,ub):
    """
    get denisty corresponding to upper bounds for Gaussian quadrature density
    
    Parameters
    ----------
    nodes : (k+2,) ndarray
    weights : (k+2,) ndarray
    
    Returns
    -------
    nodes : (k+2,) ndarray
    weights : (k+2,) ndarray
    """
    
    return np.hstack([np.nextafter(lb,-np.inf),lb,nodes[1:-1]]),\
           np.hstack([0,weights[1:-1],0])

def max_distribution(x1,y1,x2,y2):
    """
    return maximum distribution function
    
    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray
    
    Returns
    -------8.27731384e+08
    x : (k,) ndarray
    y : (k,) ndarray
    """
    
    X = np.unique(np.hstack([x1,x2]))
    Y = np.zeros_like(X)

    for i,x in enumerate(X):
        
        if x > x1[-1]:
            y1_candidate = y1[-1]
        elif x < x1[0]:
            y1_candidate = -np.inf
        else:
            y1_candidate = y1[np.argmin(x1<=x)-1]
            
        if x > x2[-1]:
            y2_candidate = y2[-1]
        elif x < x2[0]:
            y2_candidate = -np.inf
        else:
            y2_candidate = y2[np.argmin(x2<=x)-1]
            
        Y[i] = np.max([y1_candidate,y2_candidate])
        
    return X,Y

def min_distribution(x1,y1,x2,y2):
    """
    return maximum distribution function
    
    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray
    
    Returns
    -------
    x : (k,) ndarray
    y : (k,) ndarray
    """
    
    X = np.unique(np.hstack([x1,x2]))
    Y = np.zeros_like(X)

    for i,x in enumerate(X):
        
        
        if x > x1[-1]:
            y1_candidate = y1[-1]
        elif x < x1[0]:
            y1_candidate = 0
        else:
            y1_candidate = y1[np.argmin(x1<=x)-1]
            
        if x > x2[-1]:
            y2_candidate = y2[-1]
        elif x < x2[0]:
            y2_candidate = 0
        else:
            y2_candidate = y2[np.argmin(x2<=x)-1]
         
        Y[i] = np.min([y1_candidate,y2_candidate])
        
    return X,Y

def d_KS(x1,y1,x2,y2):
    """
    return KS distance between two distributions
    
    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray
    
    Returns
    -------
    dKS
    """
    
    _,ymin = min_distribution(x1,y1,x2,y2)
    _,ymax = max_distribution(x1,y1,x2,y2)
    
    return np.max(ymax-ymin)

def d_W(x1,y1,x2,y2):
    """
    return Wasserstein-1 distance between two distributions over interval a,b
    
    Parameters
    ----------
    x1 : (k,) ndarray
    y1 : (k,) ndarray
    x2 : (k,) ndarray
    y2 : (k,) ndarray
    
    Returns
    -------
    dW
    """
    
    x,ymin = min_distribution(x1,y1,x2,y2)
    _,ymax = max_distribution(x1,y1,x2,y2)
    
    return np.sum(np.diff(x)*(ymax-ymin)[:-1])