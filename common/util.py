"""Implements common utility functions."""
import logging
import time
from typing import Iterable, Tuple

import numpy as np


Tensor = np.ndarray


def crandn(shape: Tuple[int]) -> Tensor:
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.

    Parameters
    ----------
    size: int
        The number of samples to draw.

    Returns
    -------
    Tensor
        The samples.
    """
    # 1/sqrt(2) is a normalization factor
    return (
        np.random.standard_normal(shape) + 1j * np.random.standard_normal(shape)
    ) / np.sqrt(2)


def random_unitary(size: int) -> Tensor:
    """
    Construct a random unitary matrix of size n x n.

    More subtle issues are discussed at
    https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy

    Parameters
    ----------
    size: int
        The dimension the matrix.

    Returns
    -------
    Tensor
        The random unitary.
    """
    return np.linalg.qr(crandn((size, size)))[0]


def retained_bond_indices(svalues: Tensor, tol: float) -> Tensor:
    """
    Indices of retained singular values based on given tolerance.

    Parameters
    ----------
    svalues: Tensor
        1D array of singular values.
    tol: float
        Truncation tolerance.

    Returns
    -------
    Tensor
        The modified array of singular values.
    """
    norm = np.linalg.norm(svalues)
    if norm == 0:
        return np.array([], dtype=np.int64)
    # normalized squares
    svalues = (svalues / norm) ** 2
    # accumulate values from smallest to largest
    sort_idx = np.argsort(svalues)
    svalues[sort_idx] = np.cumsum(svalues[sort_idx])
    return np.where(svalues > tol)[0]


def split_matrix_svd(matrix: Tensor, tol: float) -> Tuple[Tensor]:
    """
    Split a matrix by singular value decomposition,
    and truncate small singular values based on tolerance.

    Parameters
    ----------
    matrix: Tensor
        2D array.
    tol: float
        Truncation tolerance.

    Returns
    -------
    Tuple[Tensor]
        The split matrices U @ S @ V^T = A.
    """
    assert matrix.ndim == 2
    try:
        timing = time.perf_counter()
        u_matrix, singular_values, v_matrix = np.linalg.svd(matrix, full_matrices=False)
        logging.debug(
            "SVD on matrix of shape %s took %f seconds",
            matrix.shape,
            time.perf_counter() - timing,
        )
    except np.linalg.LinAlgError:
        # Some instances produced an error from the underlying library
        # SVD not converging due to NaN Values.
        # Could not reproduce the problem reliably.
        logging.error("SVD error: %s", matrix.shape)
        np.nan_to_num(matrix, copy=False)
        u_matrix, singular_values, v_matrix = np.linalg.svd(matrix, full_matrices=False)
    # truncate small singular values
    idx = retained_bond_indices(singular_values, tol)
    u_matrix = u_matrix[:, idx]
    v_matrix = v_matrix[idx, :]
    singular_values = singular_values[idx]
    return u_matrix, singular_values, v_matrix


def dim_product(shape: Iterable[int]) -> float:
    """
    Same as np.ndarray.size for large tensors and pesudo tensors.

    Parameters
    ----------
    shape: Tuple[int]
       Dimension sizes.

    Returns
    -------
    float
        The number of entries.
    """
    return float(np.prod(tuple(float(x) for x in shape if x > 1), dtype=np.float128))
