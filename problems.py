from utils import lmo_fro, lmo_spectral, prox_l1, grad_gb, prox_mcp, spectral_prox_l1
from BCD import load_dataset, Hadamard_BCD

import numpy as np
import matplotlib.pyplot as plt

import torch

from tqdm import tqdm

def generateWH(m = 100, n = 100, r = 10):
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    return W, H

def update(
    g,
    lmo,
    x : np.ndarray,
    gamma,
    ):
    # gt = grad_F(xt)
    d = lmo(g)
    return x + gamma * d

def run_cvxMoreauNSD(
    Y,
    r : int,
    prox = prox_l1,
    lmo = lmo_spectral,
    beta = 1.,
    e = 1e-3,
    max_iter = 1_000,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    beta_t = beta/np.sqrt(2)
    gamma_t = 1
    WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        g_W = -D@H.T + grad_g(W, beta_t)
        W = update(g_W, lambda x : lmo(x, 1), W, gamma_t)
        g_H = -W.T@D
        H = update(g_H, lambda x : lmo(x, 1), H, gamma_t)
        WHs.append((W, H))
        
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / np.sqrt(t+2)
        gamma_t = 2 / (t+2)
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, dist_W_prox, WHs

def run_MoreauNSD(
    Y,
    r : int,
    prox = prox_l1,
    lmo = lmo_spectral,
    gamma = 1.,
    beta = 1.,
    p = 2/3,
    q = 1/4,
    e = 1e-3,
    max_iter = 1_000,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    beta_t = beta
    gamma_t = gamma
    WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        g_W = -D@H.T + grad_g(W, beta_t)
        g_H = -W.T@D
        W = update(g_W, lambda x : lmo(x, 1), W, gamma_t)
        H = update(g_H, lambda x : lmo(x, 1), H, gamma_t)
        WHs.append((W, H))
        
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / (t+1)**q
        gamma_t = gamma / (t+1)**p
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, dist_W_prox, WHs

def run_VS(
    Y,
    r : int,
    prox = prox_l1,
    L_gradf = 2.,
    beta = 1.,
    e = 1e-3,
    max_iter = 1_000,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    WHs = [(W, H)]
    
    beta_t = beta
    norm_T2 = 1 #np.max(W.shape)**2
    # gamma_t = 1/(L_gradf + norm_T2/beta_t)
    # gamma_tW = 1/(L_gradf*np.linalg.norm(H)**2 + norm_T2/beta_t)
    # gamma_tH = 1/(L_gradf*np.linalg.norm(W)**2 + norm_T2/beta_t)
    gamma_tW = 1/(L_gradf + norm_T2/beta_t)
    gamma_tH = 1/(L_gradf + norm_T2/beta_t)
    
    loss = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        g_W = -D@H.T + grad_g(W, beta_t)
        g_H = -W.T@D

        W = update(g_W, lambda x : lmo_fro(x, 1), W, gamma_tW)
        H = update(g_H, lambda x : lmo_fro(x, 1), H, gamma_tH)
        
        WHs.append((W, H))
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / (t+1)**(1/3)
        # gamma_t = 1/(L_gradf + norm_T2/beta_t)
        gamma_tW = 1/(L_gradf + norm_T2/beta_t)
        gamma_tH = 1/(L_gradf + norm_T2/beta_t)
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, dist_W_prox, WHs


def run_subgradient_descent(
    Y,
    g,
    shapes,
    init='random',
    # step_size_rule=lambda k: 1.0 / (k + 1)**(1/4),
    step_size_rule=1e-3,
    max_iter=1000,
    tol=1e-6,
    callback=None,
    device='cpu',
    dtype=torch.float32,
    seed=None
):
    """
    Subgradient steepest descent for matrix factorization problems F(W, H).
    
    Parameters:
    -----------
    f : callable
        Objective function F(W, H) that accepts two PyTorch tensors and returns a scalar tensor
    shapes : tuple of tuples
        (shape_W, shape_H) where shape_W = (n_rows_W, n_cols_W), shape_H = (n_rows_H, n_cols_H)
    init : str or callable, optional
        Initialization method for W and H:
        - 'random': Normal distribution (mean=0, std=0.1)
        - 'uniform': Uniform distribution [0, 1)
        - 'zeros': Zero matrices
        - callable: Custom init function taking (shape_W, shape_H) and returning (W0, H0)
    step_size_rule : callable or float, optional
        Step size rule (constant or function of iteration index k)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping based on subgradient norm
    callback : callable, optional
        Function called after each iteration with signature:
        callback(iteration, W, H, f_val, grad_norm)
    device : str or torch.device, optional
        Device to perform computations ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for matrices
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    W : torch.Tensor
        Optimized W matrix
    H : torch.Tensor
        Optimized H matrix
    history : dict
        Optimization history containing:
        - 'f': list of function values
        - 'grad_norm': list of combined gradient norms
        - 'W': list of W matrices (if callback stores them)
        - 'H': list of H matrices (if callback stores them)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    # Initialize matrices
    shape_W, shape_H = shapes
    if callable(init):
        W, H = init(shape_W, shape_H)
    elif init == 'random':
        W = 0.1 * torch.randn(*shape_W, device=device, dtype=dtype)
        H = 0.1 * torch.randn(*shape_H, device=device, dtype=dtype)
    elif init == 'uniform':
        W = torch.rand(*shape_W, device=device, dtype=dtype)
        H = torch.rand(*shape_H, device=device, dtype=dtype)
    elif init == 'zeros':
        W = torch.zeros(*shape_W, device=device, dtype=dtype)
        H = torch.zeros(*shape_H, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported initialization method: {init}")
    
    # Enable gradient tracking
    W = W.detach().requires_grad_(True)
    H = H.detach().requires_grad_(True)
    
    # History tracking
    loss = np.zeros(max_iter)
    WHs = [(W.detach().numpy(), H.detach().numpy())]
    
    Y = torch.from_numpy(Y).to(device, dtype)
    
    # history = {'f': [], 'grad_norm': [], 'WH' : [(W.item(), H.item())]}
    
    for k in range(max_iter):
        # Compute function value and gradients
        # f_val = f(W, H)
        D = (Y - W@H)
        f_val = torch.norm(D, 'fro')**2
        loss[k] = f_val.item()
        f_val += g(W)
        f_val.backward()
        
        # Get gradients and compute norm
        gW = W.grad.clone()
        gH = H.grad.clone()
        grad_norm = torch.norm(torch.cat([gW.flatten(), gH.flatten()])).item()
        
        
        # Stopping criterion
        if grad_norm < tol:
            print(f"Converged at iteration {k} with gradient norm {grad_norm:.2e}")
            break
        
        # Get step size
        step = step_size_rule(k) if callable(step_size_rule) else step_size_rule
        
        # Update matrices
        with torch.no_grad():
            W_new = W - step * gW
            H_new = H - step * gH
        
        # Prepare for next iteration (break computation graph)
        W = W_new.detach().requires_grad_(True)
        H = H_new.detach().requires_grad_(True)
        
        # Store history
        WHs.append((W.detach().numpy(), H.detach().numpy()))
        
        # Optional callback
        if callback is not None:
            callback(k, W, H, f_val.item(), grad_norm)
    
    return loss, None, WHs


if __name__ == "__main__":
    # D = load_dataset("synthetic", m = 250, n = 250)
    # D = load_dataset("olivetti")
    # D = load_dataset("spectrometer")
    # D = load_dataset("football")
    # D = load_dataset("miserables")
    # D = load_dataset("low_rank_synthetic")
    W, H = generateWH()
    D = W@H
    F_min = np.linalg.norm(W, 1)
    
    print(D.shape)
    norm_D = np.linalg.norm(D, 'fro')**2
    
    K = 1_000
    rank = 10
    
    prox = spectral_prox_l1
    
    loss, dist_W_prox, WHs = run_MoreauNSD(D, rank, prox, max_iter = K)
    ls = np.zeros(K+1)
    for i, (W, H) in enumerate(WHs):
        ls[i] = np.linalg.norm(D - W@H, 'fro')**2 + np.linalg.norm(W, 1) - F_min
    # plt.semilogy(loss/norm_D)
    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'Spectral lmo', marker="v")
    plt.semilogy(ls)
    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'Spectral lmo', marker="v")
    
    # loss = run_MoreauNSD(D, 10, lmo = lmo_fro)
    # plt.semilogy(loss/norm_D, label = 'l2 lmo')
    
    loss, dist_W_prox, WHs = run_VS(D, rank, prox, max_iter = K)
    ls = np.zeros(K+1)
    for i, (W, H) in enumerate(WHs):
        ls[i] = np.linalg.norm(D - W@H, 'fro')**2 + np.linalg.norm(W, 1) - F_min
    plt.semilogy(ls)
    
    # plt.semilogy(loss/norm_D)
    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'Variable Smoothing BW', marker="o")
    
    # plt.semilogy(loss/norm_D)
    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'Spectral lmo', marker="v")
    # plt.semilogy(loss/norm_D)
    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'Spectral lmo', marker="v")
    # W1, H1, W2, H2, error, times = Hadamard_BCD(D, r=rank, maxiter= K)
    # print(len(error))
    # plt.semilogy(error)
    # plt.scatter(np.arange(len(error))[::50], error[::50], label = 'BCD', marker="^")
    
    plt.ylabel(r'$\|Y - WH\|_F^2$')
    plt.xlabel('Iterations')
    
    plt.legend()
    plt.show()
    