from PIL import Image
import numpy as np
import numpy.linalg as nl
from scipy.sparse import vstack, csc_matrix
from Task6_CG import difference_matrix_2d, blur_matrix
#from CG_algorithm import apply_C
import matplotlib.pyplot as plt
import time


def column_norms_sq(A_t):
    """d[j] = ||a_j||^2 for the stacked matrix A_t = [A; lam*L] (CSC)."""
    return np.asarray(A_t.multiply(A_t).sum(axis=0)).ravel()


def sweep_forward(ip, idx, dat, d, r, m):
    """Solve (D-E) w = r. s = sum_{k<j} w_k a_k is the running vector."""
    n = d.shape[0]
    w = np.zeros(n)
    s = np.zeros(m)
    for j in range(n):
        acc = 0.0
        for p in range(ip[j], ip[j + 1]):
            acc += dat[p] * s[idx[p]]
        w[j] = (r[j] - acc) / d[j]
        for p in range(ip[j], ip[j + 1]):
            s[idx[p]] += w[j] * dat[p]
    return w


def sweep_backward(ip, idx, dat, d, v, m):
    """Solve (D-E^T) z = v. 
    s = sum_{k>j} z_k a_k
    j runs n-1 ... 0."""
    n = d.shape[0]
    z = np.zeros(n)
    s = np.zeros(m)
    for j in range(n - 1, -1, -1):
        acc = 0.0
        for p in range(ip[j], ip[j + 1]):
            acc += dat[p] * s[idx[p]]
        z[j] = (v[j] - acc) / d[j]
        for p in range(ip[j], ip[j + 1]):
            s[idx[p]] += z[j] * dat[p]
    return z


def make_solveP(A_t):
    """Return solveP(r) = P^{-1} r with P = (D-E) D^{-1} (D-E^T)."""
    A_t = csc_matrix(A_t)
    col_ptr, row_off, data = A_t.indptr, A_t.indices, A_t.data
    m = A_t.shape[0]
    d = column_norms_sq(A_t)

    def solveP(r):
        w = sweep_forward(col_ptr, row_off, data, d, r, m)     # (D-E)^{-1} r
        v = d * w                               # multiply by D
        return sweep_backward(col_ptr, row_off, data, d, v, m) # (D-E^T)^{-1} v
    return solveP


def PCG_solver(apply_C, rhs, solveP, x0=None, tol=1e-8, max_it=100):
    """Standard PCG on C x = rhs, with z = P^{-1} r."""
    n = len(rhs)
    x = np.zeros(n) if x0 is None else x0.copy()
    r = rhs - apply_C(x)

    z = solveP(r)
    p = z.copy()
    rz = r @ z

    nb = nl.norm(rhs)
    relres = nl.norm(r) / nb
    m = 0

    while relres > tol and m < max_it:
        Ap = apply_C(p)
        alpha = rz / (p @ Ap)

        x += alpha * p
        r -= alpha * Ap

        z = solveP(r)

        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        relres = nl.norm(r) / nb
        m += 1
    flag = int(relres > tol)

    return m, x, relres, flag


if __name__ == "__main__":
    start_time =  time.perf_counter()
    img = Image.open("img-1.png").convert("L")   # "L" = grayscale
    img = img.resize((32,32), Image.LANCZOS)   # (width, height)
    X = np.asarray(img, dtype=float) / 255.0

    n1, n2 = X.shape
    n = n1 * n2
    beta, eps, sigma, lam = 0.005, 1e-3, 0.01, 0.05

    L = difference_matrix_2d(n1, n2)
    A = blur_matrix(n1, n2, beta, eps)

    x_true = X.reshape(-1, order="F")
    np.random.seed(123)
    b = A @ x_true + sigma * np.random.randn(n)

    A_t = vstack((A, lam * L), format="csc")       # only used for the preconditioner
    solveP = make_solveP(A_t)

    apply_C = lambda v: A.T @ (A @ v) + lam**2 * (L.T @ (L @ v))
    rhs = A.T @ b

    m, x, relres, flag = PCG_solver(apply_C, rhs, solveP, tol=1e-5, max_it=200)

    print("iterations:", m, " relres:", relres, " flag:", flag)
    print("||x - x_true||:", nl.norm(x - x_true))
    print("objective:", nl.norm(A @ x - b)**2 + lam**2 * nl.norm(L @ x)**2)

    fig, ax = plt.subplots()
    ax.imshow(x.reshape((n1, n2), order="F"), cmap="gray")
    ax.axis("off")
    fig.savefig("reconstruction.png", dpi=150, bbox_inches="tight")
    print(f"Elapsed time: {time.perf_counter() - start_time:.4f} seconds")