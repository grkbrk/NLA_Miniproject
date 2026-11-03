#from CG_algorithm import apply_C
from PIL import Image
import scipy.sparse as sps
from scipy.sparse import (lil_matrix, csr_matrix, tril, triu, eye, random, diags, kron, vstack, hstack)
from  scipy.sparse.linalg import spsolve_triangular as spsolve_tri
import numpy as np
from numpy import linalg as nl
from scipy import linalg as sl
from Task6_CG import difference_matrix_2d, blur_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
'''
def make_solvers(A_t, r,w ):
    ip, indx, data = A_t.indptr, A_t.indices, A_t.data
    n = A_t.shape[1]

    def solveMinv(y):
        d = np.zeros(n)
        x = np.zeros(n)
        w = np.zeros(A_t.shape[0])
        s = 0

        for j in range(n):
            non_zero_of_col_j = data[ip[j]:ip[j+1]]
            rows_of_col_j = indx[ip[j]:ip[j+1]]

            for k in range(len(non_zero_of_col_j)):
                d[j] = A_t[j,j]**2
                s = s + w[j]* A_t[:,j]
                w[j] = (r[j] - A_t[:,j] * s) / np.norm(A_t[j,j])
        '''

'''def make_solvers(A_t):
    col_ptr, row_off, data = A_t.indptr, A_t.indices, A_t.data
    n = A_t.shape[1]
    E = sps.csc_matrix((n,n))
    D = sps.csc_matrix((n,n))
    for j in range (n):
        aj = A_t[:,j]
        D[j,j] = aj.T @ aj

        for k in range(j):
            ak = A_t[:,k]
            E[j,k] = - aj.T @ ak

    return D, E'''
    '''
def solveP(A_t, r, D, E):
    #Solve Pz = r
    ##w = Forward sweep

    #v= scale by D
    #z = Backward sweep
    return z'''

def PCG_solver(A, b, D, E, x0=None, tol=1e-8, max_it=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    relres=1
    flag=0
    m=0 #Number of iterations



    z = solveP(A, r, D, E)
    p = z.copy()

    nb = nl.norm(b)

    #While relative residual > tol and num of iterations is < max_it
    while nl.norm(r)/nb > tol and m < max_it:
        Ap = A @ p
        rz = r.T @ z

        alpha = (rz) / (p.T @ Ap)

        x += alpha * p
        r -= alpha * Ap

        z = solveP(A, r, D, E)

        beta = (r.T @ z) / (rz)
        p = z + beta * p
        relres = nl.norm(r)/nb
        
        m += 1

    if m == max_it:
        flag = 1

    return m, x, relres, flag


if __name__ == "__main__":
    # ------------------ SEKTION FÖR BILDREKONSTRUKTION ------------------
    start_time =  time.perf_counter()
    img = Image.open("img-1.png").convert("L")   # "L" = grayscale
    img = img.resize((4,4), Image.LANCZOS)   # (width, height)
    X = np.asarray(img, dtype=float) / 255.0

    n1, n2 = X.shape
    n = n1 * n2
    beta = 0.005
    eps = 1e-3
    sigma = 0.01
    lam=0.05
    x0 = np.zeros(n)
    
    L_reg = difference_matrix_2d(n1, n2) 
    A = blur_matrix(n1, n2, beta, eps)
    print("blurring, done")

    #Forming regularized A
    A_tilde = vstack((A, lam * L_reg), format="csc")

    #True signal
    x_true = X.reshape(-1, order="F") 

    np.random.seed(123)
    e = sigma * np.random.randn(n)

    b = A @ x_true + e
    b_tilde = vstack((b, np.zeros(L_reg.shape[0])), format="csc")

    print("Setup, done")

    D, E = make_solvers(A_tilde)
    
    m, x, relres, flag= PCG_solver(A_tilde, b_tilde, D, E,x0,tol=1e-5,max_it=10)

    if flag == 1:
        print("Max iteration reached")
    print('m: ',m)
    print('relres: ', relres)
    print('x - x_true',nl.norm(x - x_true))

    fig, ax = plt.subplots()
    ax.imshow(x.reshape((n1, n2), order="F"), cmap="gray")
    ax.axis("off")
    fig.savefig("reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")