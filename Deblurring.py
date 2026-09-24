import numpy as np
import scipy
import gsvd

#Lambda: 0<=lb<=1
#Number of samples: n
def deblurr(lb = 0.05, nl = 0.5, n = 100, B = 1):

    #Constants
    ba = -1                          #Upper bound
    bb = 1                           #Lower bound
    t = np.linspace(ba,bb,n)          #Time array (equidistant)
    dt= t[1]-t[0]                   #Time step


    #True signal
    def xt(t):                       
        return np.exp(t)*np.sin(np.pi*t)

    x = xt(t)       #x at time t array

    #Gaussian kernel
    #B = 100   #In Gaussian kernel, choose sufficiently large. 1D
    c = 200    #Constant
    def K(s,t):     #Gaussian kernel (point spread function (PSF))
        return c*np.exp(-np.transpose(s-t) * B * (s-t)) #Or should the last s be x????

    #A = K(s_i,t_i)*dt , s = t.
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = K(t[i],t[j]) * dt


    #Error
    np.random.seed(1477)
    Ndist = np.random.normal(0, 1, n,)  #Normal distribution
    e = nl * Ndist       #Noise

    #Observed data
    b = A @ x + e 

    if lb == 0:
        x_re, res, rnk, s = scipy.linalg.lstsq(A, b)
        return t, x, x_re, b, []#Phi

    #L is fin-diff approx of first derivative
    L = np.eye(n-1,n)
    L = L-np.eye(n-1,n,k=1)

    #Reconstruct signal bu solving regularized lsq prob (1.3) using 
    #GSVD of (A,L)
    U, V, C, S, W, Winv = gsvd.gsvd(A,L)
            #A = U @ C @ Winv
            #L = V @ S @ Winv

    #From solving lsq problem using gsvd. See Task 4.
    #Qy = Nb
    Q = np.transpose(C) @ C + lb**2*(np.transpose(S) @ S)
    N = np.transpose(C) @ U


    #Filter factor
    #Finding out array length
    M = np.vstack((A,L))
    r = np.linalg.matrix_rank(M)

    #Extract diagonals (and add 0 at the end of beta)
    beta  = np.diag(S)
    alpha = np.diag(C)[:len(beta)]

    #Calculate filter factors
    r= min(r,len(alpha))

    #https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    xxxxx = (np.transpose(U) @ b)
    y = alpha[:r] / (alpha[:r]**2 + lb**2 * beta[:r]**2) * xxxxx[:r]

    x_re = W[:r,:r] @ y

    phi = alpha[:r]**2 / (alpha[:r]**2 + lb**2 * beta[:r]**2)

    #Time array, true signal, reconstructed signal, blurred noisy data, filter factor
    return t[:r], x[:r], x_re[:r], b[:r], phi