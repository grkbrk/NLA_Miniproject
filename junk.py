
#for i in range(1,n-1):
#    L[i,i] = L[i,i] * (x(t[i+1]) - x(t[i-1])) / dt    #Central difference
#L[0,0]   = L[0,0]   * (x(t[1])  - x(t[0]))  / dt      #Forward difference for first point
#L[-1,-1] = L[-1,-1] * (x(t[-1]) - x(t[-2])) / dt      #Backward for last point


#Box quadrate rule 
BQR = np.eye(n) 
BQR[0,0] = 1/2
BQR[-1,-1] = 1/2


min(np.linalg.norm(A @ x - b))