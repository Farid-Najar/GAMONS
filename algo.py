import numpy as np
from numba import njit


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