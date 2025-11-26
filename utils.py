import numpy as np
from numba import njit
from scipy.linalg import svd

import jax.numpy as jnp

@njit
def NewtonSchulz(M):
   # by @YouJiacheng (with stability loss idea from @leloykun)
   # https://twitter.com/YouJiacheng/status/1893704552689303901
   # https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / np.linalg.norm(M)
    for a, b, c in abc_list:
        A = M.T @ M
        I = np.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

#############################
## Vector lmo
#############################
@njit
def lmo_l2(g, r):
    # lmo to l2 ball of radius r
    norm = np.linalg.norm(g)
    if norm == 0:
        return np.zeros_like(g)
    return -r * g / norm

@njit
def lmo_l1(g, r):
    # lmo to l1 ball of radius r
    i = np.argmin(g)  # most negative direction
    x = np.zeros_like(g)
    x[i] = -r
    return x

@njit
def lmo_linf(g, r):
    # lmo to l-infinity ball of radius r
    return -r * np.sign(g)

#############################
## Matrix lmo
#############################

# @njit
def lmo_fro(G, r):
    # Frobenius norm ball
    norm = np.linalg.norm(G, 'fro')
    if norm == 0:
        return np.zeros_like(G)
    return -r * G / norm

# @njit
def lmo_nuclear(G, r):
    # Nuclear norm ball
    if np.allclose(G, 0):
        return np.zeros_like(G)
    U, s, Vt = svd(G, full_matrices=False) #TODO change this to a more efficient way
    u1 = U[:, 0]
    v1 = Vt[0, :]
    return -r * np.outer(u1, v1)

# @njit
def lmo_spectral(G, r):
    # Spectral norm ball
    if np.allclose(G, 0):
        return np.zeros_like(G)
    # U, _, Vt = svd(G, full_matrices=False) #TODO change this to a more efficient way
    # return -r * U @ Vt
    return -r*NewtonSchulz(G)

@njit
def lmo_entrywise_l1(G, r):
    # entrywise l1 ball
    flat_idx = np.argmin(G)  # index of most negative entry
    X = np.zeros_like(G)
    X.flat[flat_idx] = -r
    return X

@njit
def lmo_entrywise_linf(G, r):
    # entrywise l-infinity ball
    return -r * np.sign(G)


#############################
## Functions
#############################
# ---- MCP penalty function ----
def mcp(x, lam = 1., mu = 3.):
    """MCP"""
    x = np.asarray(x)
    p = np.zeros_like(x)
    mask1 = np.abs(x) <= mu * lam
    mask2 = ~mask1
    # Region 1: |x| <= mu*lam
    p[mask1] = lam * np.abs(x[mask1]) - (x[mask1]**2) / (2 * mu)
    # Region 2: |x| > mu*lam
    p[mask2] = 0.5 * mu * lam**2
    return p

def mcp_torch(x, lam=1.0, mu=3.0):
    """MCP penalty (torch). Mirrors numpy mcp() and supports autograd."""
    import torch

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    lam_t = torch.as_tensor(lam, dtype=t.dtype, device=t.device)
    mu_t = torch.as_tensor(mu, dtype=t.dtype, device=t.device)

    mask = t.abs() <= mu_t * lam_t
    region1 = lam_t * t.abs() - (t ** 2) / (2 * mu_t)
    region2 = 0.5 * mu_t * (lam_t ** 2)

    return torch.where(mask, region1, region2)

def mcp_jnp(x, lam = 1., mu = 3.):
    """MCP"""
    x = jnp.asarray(x)
    p = jnp.zeros_like(x)
    x_mask1 = jnp.where(jnp.abs(x) <= mu * lam, p, 0)
    # mask2 = ~mask1
    # Region 1: |x| <= mu*lam
    # p[mask1] = lam * jnp.abs(x[mask1]) - (x[mask1]**2) / (2 * mu)
    p = jnp.where(jnp.abs(x) <= mu * lam, lam * jnp.abs(x_mask1) - (x_mask1**2) / (2 * mu), p)
    # Region 2: |x| > mu*lam
    # p[mask2] = 0.5 * mu * lam**2
    p = jnp.where(jnp.abs(x) > mu * lam, 0.5 * mu * lam**2, p)
    
    
    return p
#############################
## Proximals
#############################

def prox_l1(x, b):
    return np.sign(x)*np.maximum(np.abs(x)-b, 0)

# @njit
def spectral_prox_l1_old(W, b):
    O = NewtonSchulz(W)
    res = .5*(
        (W - b*O)@O.T
        @(
            O + 
            NewtonSchulz(W@O.T@O - b*O)
        )
    )
    return res

def spectral_prox_l1(W, b):
    O = NewtonSchulz(W)
    res = .5*(
        (W - b*O)@O.T
        @(
            O + 
            NewtonSchulz(W@O.T@O - b*O)
        )
    )
    return res

# @njit
def prox_mcp(x, b, lam = 1., gamma = 3.):
    if gamma <= b:
        raise ValueError("Need gamma > t")
    x = np.asarray(x)
    beta = np.zeros_like(x)
    lower = b * lam
    upper = gamma * lam
    mask2 = (np.abs(x) > lower) & (np.abs(x) <= upper)
    mask3 = np.abs(x) > upper
    soft = np.sign(x[mask2]) * (np.abs(x[mask2]) - lower)
    beta[mask2] = soft / (1 - b / gamma)
    beta[mask3] = x[mask3]
    return beta

def grad_gb(x, prox, b):
    return (x - prox(x, b))/b
