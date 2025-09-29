""" Reusable functions

This module contains different reusable functions for calculations

"""

import numpy as np
import math
from numpy import linalg as LA
import sys
import time

# SET THE VALUES FOR THE SPOQ PARAMETERS
alpha = 7e-7
beta = 1e-3
eta = 1e-1
p = 0.5
q = 2
 

def ComputeLipschitz(N):    
    """

    Parameters
    ----------
    alpha : float
    beta : float 
    eta : float 
    p : float
    q : float 
    N : int
    
    Returns
    -------
    float
        

    """
    L1 = p * (alpha**(p - 2)) / beta**p
    L2 = p / (2 * alpha**2) * max(1, (N * alpha ** p / beta**p)**2)
    L3 = (q - 1) / eta**2
    return L1 + L2 + L3


def Lpsmooth(x):
    """This function computes the smooth Lp norm of the vector x

    Parameters
    ----------
    x : array
    alpha : float 
    p : float

    Returns
    -------
    float
        smooth Lp norm of x
    """
    res = np.sum((x**2 + alpha**2) ** (p / 2) - alpha**p)
    res = res**(1 / p)
    return res


def Lqsmooth(x):
    """This function computes the smooth Lq norm of the vector x

    Parameters
    ----------
    x : array
    mu : float 
    q : float

    Returns
    -------
    float
        smooth Lq norm of x
    """
    res = eta**q + np.sum(np.abs(x)**q)
    res = res**(1 / q)
    return res


def Fcost(x):
    """

    Parameters
    ----------
    x : array
    alpha : float 
    beta : float
    mu : float
    p : float
    q : float

    Returns
    -------
    float
    """
    lp = (np.sum((x**2 + alpha**2)**(p / 2)) - alpha**p)**(1 / p)
    lq = (eta**q + np.sum(np.abs(x)**q))**(1 / q)
    fcost = math.log(((lp**p + beta**p)**(1 / p)) / lq)
    return fcost


def condlplq(x, alpha, beta, eta, p, q, ro):
    """This function computes the metric matrix for the variable metric Forward-Backward algorithm

    Parameters
    ----------
    x : array
    alpha : float 
    beta : float
    eta : float
    p : float
    q : float
    ro : float

    Returns
    -------
    array
        metric matrix 

    """
    lp = Lpsmooth(x, alpha, p)
    Xpq = (q - 1) / ((eta**q + ro**q)**(2 / q))
    A = Xpq + (1 / (lp ** p + beta**p)) * ((x**2 + alpha**2)**(p / 2 - 1))
    return A


def gradlplq(x):
    """This function computes the gradient of smooth lp over lq function

    Parameters
    ----------
    x : array
    alpha : float 
    beta : float
    mu : float
    p : float
    q : float

    Returns
    -------
    array
        gradient of smooth lp over lq function
    """
    lp = Lpsmooth(x)
    lq = Lqsmooth(x)
    grad1 = x * ((x**2 + alpha**2)**(p / 2 - 1)) / (lp ** p + beta**p)
    grad2 = np.sign(x) * (np.abs(x)**(q - 1)) / (lq ** q)
    return grad1 - grad2

def rho_0(w):

    return np.linalg.norm(w,ord=q)

def is_in_l_q_ball(z, rho):

    return np.linalg.norm(z,ord=q) < rho 

def A_tr(w,rho):
    
    lp = Lpsmooth(w)
    khi = (q-1)/((eta**q + rho**q)**(2/q))
    A_1 = khi * np.eye(np.size(w))
    A_2 = (1/(lp**p + beta**p)) * np.diag((w**2 + alpha**2)**(p/2 - 1))
    
    return A_1 + A_2

# we define the functions and their gradients 
def Psi_np(w):
    """
    Pénalité SPOQ, avec les valeurs par défaut proposées dans le papier
    """
    lp = (np.sum((w**2 + alpha**2)**(p/2) - alpha**p))**(1/p)
    lq = (eta**q + np.sum(np.abs(w)**q))**(1/q)
    fcost = np.log(((lp**p + beta**p)**(1/p)) / lq)
    return fcost

def Phi(w,X,y):
    n = X.shape[0]
    return 0.5*np.linalg.norm(y - X @ w,ord=2)

def grad_Phi(w,X,y):
    n = X.shape[0]
    X_t = np.transpose(X)
    return X_t @ X @ w - X_t @ y

def grad_Psi(w):
    return gradlplq(w)

def F(w,X,y,lambda_pen):
    return Phi(w,X,y) + lambda_pen*Psi_np(w)

def grad_F(w,X,y,lambda_pen = 1):
    return grad_Phi(w,X,y) + lambda_pen*grad_Psi(w)


def soft_thresholding(w, threshold):
    """
    Applies the soft-thresholding operator element-wise.
    Used to promote sparsity in the L1-regularized solution.
    """
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)

def LASSO(w,X,y,lambda_pen):
    
    return Phi(w,X,y) + lambda_pen*np.linalg.norm(w,ord=1)

def scad_penalty(w, lam, a=3.7):

    """
    Computes the SCAD (Smoothly Clipped Absolute Deviation) penalty for each element in w.
    
    The SCAD penalty encourages sparsity while reducing bias for large coefficients.
    It is defined piecewise with three regions, depending on the magnitude of |w|:
    
        Region 1:      |w| <= lam
        Region 2:      lam < |w| <= a * lam
        Region 3:      |w| > a * lam

    Parameters
    ----------
    w : np.ndarray
        Input weights (can be a vector or matrix).
    lam : float
        Regularization parameter (lambda > 0).
    a : float
        SCAD parameter controlling the non-convexity (must be > 2, default is 3.7).

    Returns
    -------
    penalty : np.ndarray
        SCAD penalty applied element-wise to w.
    """
        
    abs_w = np.abs(w)
    penalty = np.zeros_like(w)
    
    mask1 = abs_w <= lam
    mask2 = (abs_w > lam) & (abs_w <= a * lam)
    mask3 = abs_w > a * lam

    # Region 1
    penalty[mask1] = lam * abs_w[mask1]

    # Region 2
    penalty[mask2] = (
        - (lam**2 - 2 * a * lam * abs_w[mask2] + abs_w[mask2]**2)
        / (2 * (a - 1))
    )

    # Region 3
    penalty[mask3] = (lam**2 * (a + 1)) / 2

    return penalty


def scad_loss(w, X, y, lambda_pen, a=3.7):

    penalty = np.sum(scad_penalty(w[1:], lambda_pen, a))  # exclude bias term

    loss = Phi(w, X, y) + penalty

    return loss 


def scad_prox(w, lambda_, a=3.7):
    """
    Proximal operator for the SCAD penalty (element-wise).
    
    Parameters:
        w : numpy array — input weights
        lambda_ : float — regularization parameter
        a : float — SCAD parameter (usually 3.7)
        
    Returns:
        numpy array — result after applying SCAD proximal operator
    """
    w_abs = np.abs(w)
    prox = np.zeros_like(w)

    # Region 1: |w| <= 2λ
    mask1 = w_abs <= 2 * lambda_
    prox[mask1] = np.sign(w[mask1]) * np.maximum(w_abs[mask1] - lambda_, 0)

    # Region 2: 2λ < |w| <= aλ
    mask2 = (w_abs > 2 * lambda_) & (w_abs <= a * lambda_)
    prox[mask2] = ((a - 1) * w[mask2] - np.sign(w[mask2]) * a * lambda_) / (a - 2)

    # Region 3: |w| > aλ → unchanged
    mask3 = w_abs > a * lambda_
    prox[mask3] = w[mask3]

    return prox


# use this function if you want to tune the parameter a in SCAD
def prox_SCAD(x, gamma, a, lamb):
    """
    Compute the proximity operator of the SCAD penalty.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    gamma : float or np.ndarray
        Positive scalar or array (same size as x).
    a : float or np.ndarray
        Parameter > 2, scalar or array (same size as x).
    lamb : float or np.ndarray
        Positive scalar or array (same size as x).

    Returns
    -------
    p : np.ndarray
        Proximal operator applied element-wise to x.
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    a = np.asarray(a)
    lamb = np.asarray(lamb)

    # Input checks
    if np.any(gamma <= 0):
        raise ValueError("'gamma' must be strictly positive")
    if np.any(a <= 2):
        raise ValueError("'a' must be strictly greater than 2")
    if np.any(lamb <= 0):
        raise ValueError("'lamb' must be strictly positive")

    abs_x = np.abs(x)
    sign_x = np.sign(x)

    # First proximal formula (p1)
    cond1 = abs_x <= (1 + gamma) * lamb
    cond2 = (abs_x > (1 + gamma) * lamb) & (abs_x <= a * lamb)
    cond3 = abs_x > a * lamb

    p1 = np.zeros_like(x, dtype=float)
    p1[cond1] = sign_x[cond1] * np.maximum(abs_x[cond1] - lamb * gamma, 0)
    p1[cond2] = ((a - 1) * x[cond2] - sign_x[cond2] * a * lamb * gamma) / (a - 1 - gamma)
    p1[cond3] = x[cond3]

    # Second proximal formula (p2)
    cond_p2_1 = abs_x <= 0.5 * (a + 1 + gamma) * lamb
    cond_p2_2 = abs_x > 0.5 * (a + 1 + gamma) * lamb

    p2 = np.zeros_like(x, dtype=float)
    p2[cond_p2_1] = sign_x[cond_p2_1] * np.maximum(abs_x[cond_p2_1] - lamb * gamma, 0)
    p2[cond_p2_2] = x[cond_p2_2]

    # Decide which formula to use depending on condition a > (1 + gamma)
    if np.isscalar(a) and np.isscalar(gamma):
        if a > (1 + gamma):
            p = p1
        else:
            p = p2
    else:
        # For arrays, select element-wise
        idx = a > (1 + gamma)
        p = np.empty_like(x, dtype=float)
        p[idx] = p1[idx]
        p[~idx] = p2[~idx]

    return p
