#MINRES
import numpy as np
from CG_algorithm import *
import time
'''
def minres(A, b, x0, tol=1e-5, max_iter=100):
    x = x0.copy()
    r = b - A @ x
    rho0 = np.linalg.norm(r)

    if rho0 == 0:
        return x, 0, 0.0, 0

    bnorm = np.linalg.norm(b)
    relres = rho0 / bnorm if bnorm != 0 else rho0

    v0 = np.zeros_like(b, dtype=float)
    v = r / rho0
    beta0 = 0.0
    p2 = np.zeros_like(b, dtype=float)
    p1 = np.zeros_like(b, dtype=float)

    # Initial MINRES rotation values
    dbar = epsln = 0.0
    phibar = rho0
    c_prev, s_prev = -1.0, 0.0

    m = 1
    flag = 1

    while relres > tol and m <= max_iter:

        # Lanczos process
        w = A @ v - beta0 * v0
        alpha = np.dot(v, w)
        w = w - alpha * v
        beta = np.linalg.norm(w)

        # Apply previous Givens rotations
        delta = c_prev * dbar + s_prev * alpha
        gamma_bar = s_prev * dbar - c_prev * alpha
        epsln_new = s_prev * beta
        dbar_new = -c_prev * beta

        # New Givens rotation
        gamma = np.hypot(gamma_bar, beta)
        if gamma == 0:
            break

        c = gamma_bar / gamma
        s = beta / gamma

        # Update search direction and solution
        p = (v - delta * p1 - epsln * p2) / gamma
        mu = c * phibar
        x = x + mu * p

        # Update residual and relative residual
        phibar = s * phibar
        relres = abs(phibar) / bnorm if bnorm != 0 else np.linalg.norm(b - A @ x)

        # Prepare next iteration
        p2, p1 = p1.copy(), p.copy()
        v0, beta0 = v.copy(), beta
        dbar, epsln = dbar_new, epsln_new
        c_prev, s_prev = c, s

        if beta == 0:
            break

        v = w / beta
        m += 1

    iterations = m - 1
    relres = np.linalg.norm(b - A @ x) / bnorm if bnorm != 0 else np.linalg.norm(b - A @ x)
    flag = 0 if relres <= tol else 1

    return x, iterations, relres, flag
    '''
def minres(C_op, D, x0, A, L, b, x_true, lam, tol=1e-5, max_iter=100):
    x = x0.copy()
    
    # Förbered listor för att spara historik för varje iteration
    obj_history = []
    rel_res_history = []
    rre_history = []

    norm_D = np.linalg.norm(D)
    norm_xtrue = np.linalg.norm(x_true)

    # Initial residual for MINRES system (C_op * x0 - D, or b - A*x0 depending on formulation)
    r = D - C_op @ x0
    rho0 = np.linalg.norm(r)

    if rho0 == 0:
        # Sök igenom första steget om initial residual är noll
        Ax = A @ x
        Lx = L @ x
        obj_history.append(np.linalg.norm(Ax - b) ** 2 + (lam ** 2) * np.linalg.norm(Lx) ** 2)
        rel_res_history.append(0.0)
        rre_history.append(np.linalg.norm(x_true - x) / norm_xtrue)
        return x, 0, r, obj_history, rel_res_history, rre_history

    bnorm = norm_D
    relres = rho0 / bnorm if bnorm != 0 else rho0

    v0 = np.zeros_like(D, dtype=float)
    v = r / rho0
    beta0 = 0.0
    p2 = np.zeros_like(D, dtype=float)
    p1 = np.zeros_like(D, dtype=float)

    # Initial MINRES rotation values
    dbar = epsln = 0.0
    phibar = rho0
    c_prev, s_prev = -1.0, 0.0

    m = 1
    while relres > tol and m <= max_iter:
        # --- BERÄKNA METRIKER FÖR DENNA ITERATION ---
        Ax = A @ x
        Lx = L @ x
        obj_history.append(np.linalg.norm(Ax - b) ** 2 + (lam ** 2) * np.linalg.norm(Lx) ** 2)
        
        # Relativ residual för de normala ekvationerna
        r_normal = D - C_op @ x
        rel_res_history.append(np.linalg.norm(r_normal) / norm_D)
        
        # Relativt rekonstruktionsfel (RRE)
        rre_history.append(np.linalg.norm(x_true - x) / norm_xtrue)

        # Lanczos process (använder C_op istället för A)
        w = C_op @ v - beta0 * v0
        alpha = np.dot(v, w)
        w = w - alpha * v
        beta = np.linalg.norm(w)

        # Apply previous Givens rotations
        delta = c_prev * dbar + s_prev * alpha
        gamma_bar = s_prev * dbar - c_prev * alpha
        epsln_new = s_prev * beta
        dbar_new = -c_prev * beta

        # New Givens rotation
        gamma = np.hypot(gamma_bar, beta)
        if gamma == 0:
            break

        c = gamma_bar / gamma
        s = beta / gamma

        # Update search direction and solution
        p = (v - delta * p1 - epsln * p2) / gamma
        mu = c * phibar
        x = x + mu * p

        # Update residual and relative residual
        phibar = s * phibar
        relres = abs(phibar) / bnorm if bnorm != 0 else np.linalg.norm(D - C_op @ x) / norm_D

        # Prepare next iteration
        p2, p1 = p1.copy(), p.copy()
        v0, beta0 = v.copy(), beta
        dbar, epsln = dbar_new, epsln_new
        c_prev, s_prev = c, s

        if beta == 0:
            break

        v = w / beta
        m += 1

    iterations = m - 1
    residual = D - C_op @ x
    
    return x, iterations, residual, obj_history, rel_res_history, rre_history

if __name__ == "__main__":
    #Setting start variables
    img = Image.open("img-1.png").convert("L")   # "L" = grayscale
    X = np.asarray(img, dtype=float) / 255.0

    n1, n2 = X.shape
    n = n1 * n2
    beta = 0.05
    eps = 1e-3
    sigma = 0.01
    x0 = np.zeros(n)
    
    L = mp.difference_matrix_2d(n1, n2) 
    A = mp.blur_matrix(n1, n2, beta, eps)

    x_true = X.reshape(-1, order="F") 
    np.random.seed(123)
    e = sigma * np.random.randn(n)
    b = A @ x_true + e 
    B = b.reshape((n1, n2), order="F") 

    # --- REGREGLAGE (Dessa kan du ändra på för dina experiment!) ---

    def apply_C(v, lam):
        return A.T @ (A @ v) + lam**2 * (L.T @ (L @ v))

    C = LinearOperator((n, n), matvec=apply_C, dtype=float)
    D = A.T @ b

    '''
    #Now comparing CG method and minres method
    lam = 0.1
    stopping_tol = 1e-7
    max_iterations = 1000

    C = LinearOperator((n, n), matvec=lambda v, lam=lam: apply_C(v, lam),dtype=float)
    X, iterations, residual, obj_hist, rel_res_hist, rre_hist = CG(C, D, x0, A, L, b, x_true, lam, tol=stopping_tol, max_iter=max_iterations)
    X_rec = X.reshape((n1, n2), order="F")
    
    x, iterations, relres, flag = minres(A, b, x0, tol = stopping_tol, max_iter = max_iterations)
    x_rec = x.reshape((n1, n2), order="F")

    lst = [X_rec, x_rec]
    values = [1,1] #random numbers. I don't care
    obj = [obj_hist] #ADD HERE
    rel = [rel_res_hist] #ADD HERE
    rre = [rre_hist] #ADD HERE
    
    plot_evolution(lst, values, title=f'Varying Stopping Tolerance')
    plot_convergence(obj, rel, rre, title=f'Convergence Plots for Varying Stopping Tolerance')
    '''
# --- JÄMFÖRELSE MELLAN CG OCH MINRES ---
    lam = 0.1
    stopping_tol = 1e-7
    max_iterations = 40 #Changing this so we see a difference when plotting images

    C = LinearOperator((n, n), matvec=lambda v, lam=lam: apply_C(v, lam), dtype=float)
    
    # Kör CG
    start_cg = time.time()
    X_cg, it_cg, res_cg, obj_cg, rel_cg, rre_cg = CG(C, D, x0, A, L, b, x_true, lam, tol=stopping_tol, max_iter=max_iterations)
    end_cg = time.time()
    print("Elapsed time CG: ", end_cg - start_cg)
    X_rec_cg = X_cg.reshape((n1, n2), order="F")
    
    # Kör MINRES (med samma C och D)
    start_cg = time.time()
    X_mr, it_mr, res_mr, obj_mr, rel_mr, rre_mr = minres(C, D, x0, A, L, b, x_true, lam, tol=stopping_tol, max_iter=max_iterations)
    end_cg = time.time()
    print("Elapsed time minres: ", end_cg - start_cg)
    X_rec_mr = X_mr.reshape((n1, n2), order="F")

    # Samla för plotting (a = CG, b = MINRES)
    lst = [X_rec_cg, X_rec_mr]
    values = [f'CG (it: {it_cg})', f'MINRES (it: {it_mr})']
    
    obj = [obj_cg, obj_mr]
    rel = [rel_cg, rel_mr]
    rre = [rre_cg, rre_mr]
    
    plot_evolution(lst, values, title='Comparison: CG vs MINRES')
    plot_convergence(obj, rel, rre, title='Convergence Plots: CG vs MINRES')
    
