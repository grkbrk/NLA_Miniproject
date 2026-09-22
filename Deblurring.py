import numpy as np;
import math;

a = -1                          #Upper bound
b = 1                           #Lower bound
n = 10                          #Number of samples
t = np.linspace(a,b,n)          #Time array (equidistant)
dt= t[1]-t[0]                   #Time step
x = np.exp(t)*np.sin(np.pi*t)   #x at time t array

#True signal
def xt(t):                       
    return np.exp(t)*np.sin(np.pi*t)

#Gaussian kernel
B = 1   #In Gaussian kernel, choose sufficiently large. 1D
c = 1   #Const, which value to use?
def K(s,t):     #Gaussian kernel (point spread function (PSF))
    return c*np.exp(-np.transpose(s-t) @ B @ (s-t)) #Or should the last s be x????

#A = K(s_i,t_i)*dt , s = t.
A = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        A[i,j] = K(t[i],t[j]) * dt

#Error
Ndist = np.random.normal(0, 1, n)  #Normal distribution
nl = 0.5            #Noise level
e = nl * Ndist       #Noise

#Observed data
b = A @ x + e    

#Reconstruct signal bu solving regularized lsq prob (1.3) using GSVD of (A,L)
#where L is fin-diff approx of first derivative.
L = np.eye(n-1,n)
L = L-np.eye(n-1,n,k=1)

min(np.linalg.norm(A @ x - b))