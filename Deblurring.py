import numpy as np;
import math;

a = -1                  #Upper bound
b = 1                   #Lower bound
n = 10                  #Number of samples
t = np.linspace(a,b,n)           #Time array
dt= t[1]-t[0]              #Time step
xt = np.exp(t)*np.sin(np.pi*t)   

def x(t):               #True signal
    return np.exp(t)*np.sin(np.pi*t)

#Box quadrate rule 
A = np.eye(n) #Matrix A (deblurring operator)
A[0,0] = 1/2
A[-1,-1] = 1/2

B = 1 #In Gaussian kernel, choose sufficiently large.
c = 1 #Const, which value to use?
def K(s,t):     #Gaussian kernel
    return c*np.exp(-np.transpose(s-t) @ B @ (s-t)) #Or should the last s be x????

Ndit = np.random.normal(0, 1, n)  #Normal distribution
nl = 0.5    #Noise level
e = nl*Ndit #Noise
b = A@xt + e #Observed data

#Reconstruct signal bu solving regularized lsq prob (1.3) using GSVD of (A,L)
#where L is fin-diff approx of first derivative.
L = np.eye(n-1,n)
L = L-np.eye(n-1,n,k=1)

min(np.linalg.norm(A@xt-b))