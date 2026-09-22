import numpy as np
n = 10
L = np.eye(n-1,n)
L = L-np.eye(n-1,n,k=1)
print(L)