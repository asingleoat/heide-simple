"""
Operator norm computation for primal-dual optimization.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


def compute_operator_norm(A, AS, shape, tol=1e-3, maxiter=10):
    """
    Compute the operator norm for a linear operator.

    The operator norm is the square root of the largest eigenvalue of AS*A,
    where A is the forward operator and AS is its adjoint.

    Parameters
    ----------
    A : callable
        Forward operator: A(x) where x is an image of given shape
    AS : callable
        Adjoint operator: AS(y)
    shape : tuple
        Image shape (height, width)
    tol : float
        Tolerance for eigenvalue computation
    maxiter : int
        Maximum iterations for eigenvalue computation

    Returns
    -------
    L : float
        Operator norm (largest singular value of A)
    """
    m, n = shape
    vec_size = m * n

    def matvec(x_vec):
        """Apply AS(A(x)) to a vectorized image."""
        x_img = x_vec.reshape(m, n)
        result = AS(A(x_img))
        return result.ravel()

    # Create LinearOperator for scipy's eigsh
    op = LinearOperator((vec_size, vec_size), matvec=matvec, dtype=np.float64)

    # Compute largest eigenvalue using Arnoldi iteration
    try:
        eigenvalues, _ = eigsh(op, k=1, which='LM', tol=tol, maxiter=maxiter)
        lambda_largest = eigenvalues[0]
    except Exception:
        # Fallback to power iteration if eigsh fails
        lambda_largest = _power_iteration(matvec, vec_size, tol, maxiter * 10)

    return np.sqrt(np.abs(lambda_largest))


def _power_iteration(matvec, n, tol, maxiter):
    """
    Power iteration to find largest eigenvalue.

    Fallback method if scipy's eigsh fails.
    """
    # Random initial vector
    x = np.random.randn(n)
    x = x / np.linalg.norm(x)

    lambda_old = 0
    for _ in range(maxiter):
        # Apply operator
        y = matvec(x)

        # Rayleigh quotient for eigenvalue estimate
        lambda_new = np.dot(x, y)

        # Normalize
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-10:
            break
        x = y / norm_y

        # Check convergence
        if abs(lambda_new - lambda_old) < tol * abs(lambda_new):
            break
        lambda_old = lambda_new

    return lambda_new
