import numpy as np
from numba import njit

from utils import lmo_fro, lmo_spectral, grad_gb, prox_mcp

class GradientDescent:
    def __init__(
        self,
        gamma = 1e-3,
        lmo = None,
        ):
        """Gradient descent scheme and algorithm

        Parameters
        ----------
        gamma : float, optional
            step size (may be time dependent), by default 1e-3
        lmo : function, optional
            the linear minimization oracle function, by default None
        """
        self.gamma = gamma
        self.lmo = lmo
    
    def update(self, x, grad, gamma = None):
        if gamma is None:
            gamma = self.gamma
        g = grad(x)
            
        d = -g if self.lmo is None else self.lmo(g)
        return x + gamma * d
    
    
def _update_gamons(
    xt : np.ndarray,
    gt : np.ndarray,
    gamma : float,
    lmo : callable,
    ):
    dt = lmo(gt)
    return xt + gamma * dt

def gamons(
    x0,
    grad_f,
    T : np.ndarray,
    prox = prox_mcp,
    lmo = lmo_spectral,
    gamma = 1/3,
    beta = 1.,
    p = 2/3,
    q = 1/4,
    e = 1e-3,
    max_iter = 1_000,
    store_xt_every = 1,
    ):
    grad_g = lambda x, b : grad_gb(x, prox, b)
    
    grad_norms = np.zeros(max_iter)
        
    xt = x0
    xts = [xt]
    
    for t in tqdm(range(max_iter)):
        gt = grad_f(xt) + T.T@grad_g(T@xt, beta)
        grad_norms[t] = np.linalg.norm(gt)
        
        xt = _update_gamons(xt, gt, gamma, lmo)
        if t % store_xt_every == 0:
            xts.append(xt)
    
    return xts

@njit
def f(x : np.ndarray):
    """ The convex function Lf-smooth

    Parameters
    ----------
    x : np.ndarray
        input
    """
    pass

@njit
def g(x : np.ndarray):
    """ The (weakly)-convex function non-smooth

    Parameters
    ----------
    x : np.ndarray
        input
    """
    pass

@njit
def g_b(x : np.ndarray, b : float):
    """ Smoothed g with the Moreau envelope 1/b-smooth

    Parameters
    ----------
    x : np.ndarray
        input
    """
    pass

@njit
def T(x : np.ndarray):
    """ Linear operator

    Parameters
    ----------
    x : np.ndarray
        input
    """
    pass

F = lambda x : f(x) + g(T(x))

def update(
    grad_F,
    lmo,
    xt : np.ndarray,
    gamma,
):
    gt = grad_F(xt)
    dt = lmo(gt)
    return xt + gamma * dt