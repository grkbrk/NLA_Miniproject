import numpy as np;
import math;

a = -1                  #Upper bound
b = 1                   #Lower bound
n = 10                  #Number of samples
t = np.linspace(a,b,n)           #Time array
xt = np.exp(t)*np.sin(np.pi*t)   

def x(t):               #True signal
    return np.ext(t)*np.sin(np.pi*t)

#A = #Matrix A (deblurring operator)
#Box quadrate rule 
B = 1 #In Gaussian kernel
c = 1 #Const
def K(s,t):     #Gaussian kernel
    return c*np.exp(-np.transpose(s-t) @ B @ (x-t))

#Ndit = #Normal distribution
#del = #Noise level
#e = del*Ndist
#b = Ax + e#Observed data

#Reconstruct signal bu solving regularized lsq prob (1.3) using GSVD of (A,L)
#where L if fin-diff approx of first derivative.

min(norm2(A*x-b))