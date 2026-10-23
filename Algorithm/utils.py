import numpy as np

# naive power method
def power_method(A, iterations=1000):
    xk = np.ones((A.shape[0], 1))
    while iterations > 0:
        xk = (A @ xk)
        # scaling the result
        xk /= np.sum(xk)
        iterations -= 1
    return xk


# advanced version of power method tailored for pageRank
    # introduction of teleportation
    # convergence used as a stopping criterion
    # dangling nodes correction
def power_method_adv(A, iterations=1000):
    m = 0.15
    xk = np.ones((A.shape[0], 1))
    eig = float("inf")
    eig_old = -float("inf")

    while iterations > 0 and abs(eig - eig_old) > 1e-12:
        eig_old = eig
        xk = (1 - m) * (A @ xk) + m * np.full((A.shape[0], 1), 1 / A.shape[0])
        xk /= np.sum(xk)
        eig = (A @ xk).T @ xk
        iterations -= 1

    return xk


# implementation of matrix by vector product
# tailored for sparse crs matrix representation
def csr_vec_multiplication(LV, JC, IA, xk, dangling_nodes):
    w = np.zeros((xk.shape[0], 1))
    for i in range(len(IA) - 1):
        for k in range(IA[i], IA[i + 1]):
            w[i, 0] += LV[k] * xk[JC[k], 0]

    # add contribution of dangling nodes (see report)
    n = xk.shape[0]
    dangling_node_contribution = np.sum(xk[dangling_nodes])
    w += dangling_node_contribution / n

    return w


# advanced version of power method on sparse matrices tailored for pageRank
    # introduction of teleportation
    # convergence used as a stopping criterion
    # dangling nodes correction
def power_method_adv_csr(LV, JC, IA, tot_rows, dangling_nodes, iterations=1000, m=0.15):
    xk = np.ones((tot_rows, 1))
    eig = float("inf")
    eig_old = -float("inf")
    n_iterations_per_page = np.zeros((tot_rows), dtype=int)
    conv_rate = []
    n_iter = 0

    while iterations > 0 and abs(eig - eig_old) > 1e-12:
        n_iter += 1
        xk_prev = xk.copy()
        eig_old = eig

        intermed_res = csr_vec_multiplication(LV, JC, IA, xk, dangling_nodes)
        eig = intermed_res.T @ xk

        xk = (1 - m) * intermed_res + m * np.full((tot_rows, 1), 1 / tot_rows)
        xk /= np.sum(xk)
        iterations -= 1

        abs_diff = np.abs(xk - xk_prev).flatten()
        conv_rate.append(np.linalg.norm(abs_diff))
        new_conv = (n_iterations_per_page == 0) & (abs_diff < 1e-12)
        n_iterations_per_page[new_conv] = n_iter

    # fill remaining pages that haven't converged yet
    has_not_conv = n_iterations_per_page == 0
    n_iterations_per_page[has_not_conv] = n_iter

    return xk, n_iterations_per_page, conv_rate
