#from CG_algorithm import apply_C
from PIL import Image
from scipy.sparse import (lil_matrix, csr_matrix, tril, triu, eye, random, diags, kron)
from  scipy.sparse.linalg import spsolve_triangular as spsolve_tri
import numpy as np
from numpy import linalg as nl
from scipy import linalg as sl
from Task6_CG import difference_matrix_2d, blur_matrix
import matplotlib.pyplot as plt

def PCG_solver(A, b, solveM, x0=None, tol=1e-8, max_it=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    relres=1
    flag=0
    m=0 #Number of iterations
    z = solveM(r)
    p = z.copy()

    nb= nl.norm(b)

    while nl.norm(r)/nb > tol and m < max_it:
        Ap = A @ p
        rz = r.T @ z

        alpha = (rz) / (p.T @ Ap)

        x += alpha * p
        r_new = r - alpha * Ap

        z_new = solveM(r_new)

        beta = (r_new.T @ z_new) / (rz)
        p = z_new + beta * p
        relres = nl.norm(r_new)/nb
        
        z,r = z_new, r_new
        m += 1

    if m == max_it:
        flag = 1

    return m, x, relres, flag



def ic0(A):
    """incomplete Cholesky factorisation"""
    A = tril(csr_matrix(A, dtype=float)).tocsr()
    A.sort_indices()
    n = A.shape[0]
    rows = [dict() for _ in range(n)]

    for i in range(n):
        Li = rows[i]  #Entries in row i of L: column:value
        Li.items()
        for ind in range(A.indptr[i], A.indptr[i+1]):
            j = A.indices[ind]
            a = A.data[ind]
            if j < i:
                Lj = rows[j]
                t=0
                for k,Lik in Li.items():
                    if k in Lj:
                        t += Lik * Lj[k]
                s = a - t
                Li[j] = s / Lj[j]
            else:
                s = a - sum(v*v for v in Li.values())
                if s <= 0:
                    raise ValueError (f"Non-positive pivot at row {i}")
                Li[i] = np.sqrt(s)

    #reset
    indptr = [0]
    indices = []
    data = []
    for i in range(n):
        for c in sorted(rows[i]):
            indices.append(c)
            data.append(rows[i][c])
        indptr.append(len(indices))

    L = csr_matrix((data, indices, indptr), shape=(n,n))
    return (L)

def ic_precondition(A):
    L = ic0(A)
    def solveM (r):
        y = spsolve_tri(L,r, lower=True)
        z = spsolve_tri(L.T.tocsr(),y, lower=False)
        return z

    return solveM


if __name__ == "__main__":
    # ------------------ SEKTION FÖR BILDREKONSTRUKTION ------------------
    
    img = Image.open("img-1.png").convert("L")   # "L" = grayscale
    img = img.resize((64, 64), Image.LANCZOS)   # (width, height)
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
    
    x_true = X.reshape(-1, order="F") 

    A_normal = (A.T @ A + lam * L_reg.T @ L_reg).tocsr()
    print('A_normal found')

    np.random.seed(123)
    e = sigma * np.random.randn(n)

    b = A @ x_true + e 
    b_normal = A.T @ b

    print("Setup, done")
    M_ic  = ic_precondition(A_normal)
    print("Precondition, done")
    m, x, relres, flag= PCG_solver(A_normal,b_normal,M_ic,x0,tol=1e-5,max_it=10)

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