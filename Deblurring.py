import numpy as np
import gsvd

#Lambda: 0<=lb<=1
#Number of samples: n
def deblurr(lb = 0.5, nl = 0.5, n = 100):

    ba = -1                          #Upper bound
    bb = 1                           #Lower bound
    #n = 10                         #Number of samples
    t = np.linspace(ba,bb,n)          #Time array (equidistant)
    dt= t[1]-t[0]                   #Time step
    #x = np.exp(t)*np.sin(np.pi*t)   #x at time t array

    #True signal
    def xt(t):                       
        return np.exp(t)*np.sin(np.pi*t)

    x = xt(t)

    #Gaussian kernel
    B = 1   #In Gaussian kernel, choose sufficiently large. 1D
    c = 1   #Const, which value to use?
    def K(s,t):     #Gaussian kernel (point spread function (PSF))
        return c*np.exp(-np.transpose(s-t) * B * (s-t)) #Or should the last s be x????

    #A = K(s_i,t_i)*dt , s = t.
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = K(t[i],t[j]) * dt

    #Error
    Ndist = np.random.normal(0, 1, n)  #Normal distribution
    #nl = 0.5            #Noise level
    e = nl * Ndist       #Noise

    #Observed data
    b = A @ x + e 

    #L is fin-diff approx of first derivative
    L = np.eye(n-1,n)
    L = L-np.eye(n-1,n,k=1)

    #Reconstruct signal bu solving regularized lsq prob (1.3) using 
    #GSVD of (A,L)
    U, V, C, S, W, Winv = gsvd.gsvd(A,L)
            #A = U @ C @ Winv
            #L = V @ S @ Winv

    #print("C:\n",C.shape)
    #print("S:\n",S.shape)

    #From solving lsq problem using gsvd. See Task 4.
    #Qy = Nb
    Q = np.transpose(C) @ C + lb**2*(np.transpose(S) @ S)
    N = np.transpose(C) @ U


    #y = np.invert(M) @ N @ b
    y = (1/Q) @ N @ b

    x_re = W @ y

    #Filter factor
    #Finding out array length
    M = np.vstack((A,L))
    print(M.shape)
    r = np.linalg.matrix_rank(M)
    print("r: ",r)

    #Extract diagonals (and add 0 at the end of beta)
    alpha = np.diag(C)
    beta  = np.concatenate([np.diag(S),[0]])
    #Calculate filter factors
    Phi = alpha[:r]**2/ (alpha[:r]**2 + lb**2 * beta[:r]**2)

    #Time array, true signal, reconstructed signal, blurred noisy data, filter factor
    return t, x, x_re, b, Phi