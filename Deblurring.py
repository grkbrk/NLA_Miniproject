import numpy as np
import gsvd

#Lambda: 0<=lb<=1
#Number of samples: n
def deblurr(lb = 0.5, nl = 0.5, n = 100, B = 1):

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
    #B = 100   #In Gaussian kernel, choose sufficiently large. 1D
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

    print("A finite:", np.isfinite(A).all())
    for name, M_ in [("U", U), ("W", W), ("C", C), ("S", S)]:
        print(name, M_.shape, "finite:", np.isfinite(M_).all())

    alpha = np.diag(C)
    d = np.diag(S)
    print("alpha[:5] =", alpha[:5], " alpha[-5:] =", alpha[-5:])
    print("diag(S)[:5] =", d[:5], " diag(S)[-5:] =", d[-5:])

    # Do alpha^2 + beta^2 = 1 hold for the first n-1 pairs?
    print("max |alpha^2+beta^2-1| (first n-1):", np.abs(alpha[:len(d)]**2 + d**2 - 1).max())
    print("alpha[-1] =", alpha[-1], "(expect 1)")

    # Does the decomposition reproduce A and L?
    print("||A - U C Winv|| =", np.linalg.norm(A - U @ C @ Winv))
    print("||L - V S Winv|| =", np.linalg.norm(L - V @ S @ Winv))

    #print("C:\n",C.shape)
    #print("S:\n",S.shape)

    #From solving lsq problem using gsvd. See Task 4.
    #Qy = Nb
    Q = np.transpose(C) @ C + lb**2*(np.transpose(S) @ S)
    N = np.transpose(C) @ U


    #Filter factor
    #Finding out array length
    M = np.vstack((A,L))
    print(M.shape)
    r = np.linalg.matrix_rank(M)
    print("r: ",r)

    #Extract diagonals (and add 0 at the end of beta)
    alpha = np.diag(C)[1:]
    beta  = np.diag(S)#np.concatenate([[0],np.diag(S)])
    #Calculate filter factors
    r= min(r,len(alpha))
    Phi = alpha[:r]**2/ (alpha[:r]**2 + lb**2 * beta[:r]**2)

    
    ##y = np.invert(Q) @ N @ b
    #y = (1/Q) @ N @ b
    #https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
    xxxxx = (np.transpose(U) @ b)
    y = Phi / alpha * xxxxx[:r]

    print("y shape: ", y.shape)

    x_re = W[:r,:r] @ y
    print(np.isfinite(x_re).all())
    print(np.isnan(x_re).sum(), "NaNs in x_re")
    #x_reflat = np.concatenate([x_re.flatten, [0]])
    #print("x_re shape flat: ", np.size(x_reflat))

    #Time array, true signal, reconstructed signal, blurred noisy data, filter factor
    return t[:r], x[:r], x_re, b, Phi