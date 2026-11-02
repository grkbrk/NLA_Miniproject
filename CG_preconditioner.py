#from CG_algorithm import apply_C
from PIL import Image
from scipy.sparse import (lil_matrix, csr_matrix, tril, triu, eye, random, diags, kron, vstack, hstack)
from  scipy.sparse.linalg import spsolve_triangular as spsolve_tri
import numpy as np
from numpy import linalg as nl
from scipy import linalg as sl
from Task6_CG import difference_matrix_2d, blur_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def PCG_solver(A, b, x0=None, tol=1e-8, max_it=100):
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x
    relres=1
    flag=0
    m=0 #Number of iterations

    D = diags(A.diagonal(), 0, format="csr") 
    E = -tril(A, k=-1, format="csr")
    F = -triu(A, k=1, format="csr")
    DE = D - E
    DF = D - F

    def solveM(v):
        w = spsolve_tri(DE, v, lower=True)
        w = D @ w
        return spsolve_tri(DF, w, lower=False)

    z = solveM(r)
    p = z.copy()

    nb = nl.norm(b)

    #While relative residual > tol and num of iterations is < max_it
    while nl.norm(r)/nb > tol and m < max_it:
        Ap = A @ p
        rz = r.T @ z

        alpha = (rz) / (p.T @ Ap)

        x += alpha * p
        r -= alpha * Ap

        z = solveM(r)

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
    b_tilde = hstack((b, csr_matrix(np.zeros((b.shape[0], L_reg.shape[0])))), format="csc")

    print("Setup, done")

    
    m, x, relres, flag= PCG_solver(A_tilde,b_tilde,x0,tol=1e-5,max_it=10)

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