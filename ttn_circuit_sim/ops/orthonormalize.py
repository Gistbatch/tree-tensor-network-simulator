"""Implements normalization methods on tree nodes."""
from typing import Optional
import logging
import time

import numpy as np
import opt_einsum as oe

from common import split_matrix_svd, Tensor
from ..tnode import PseudoTNode, TNode


def orthonormalize_qr(node: TNode, i: int, factor: Optional[Tensor] = None) -> Tensor:
    """Orthonormalization using QR decomposition.

    Similar to orthonormalize_svd but direct truncation of entries is not possible.

    Parameters
    ----------
    node: TNode
        The node to normalize.
    i: int
        Index on which the normalization comes from.
    factor: Optional[Tensor], default = None
        Non-normal part from previous node.

    Returns
    -------
    Tensor
        The non-normal part of the current node.

    """
    tensor = node.tensor
    if factor is not None:  # absorb the result from lower level normalization
        tensor = contract_factor_on_index(tensor, factor, node.leaf_indices[i])
    shape = tensor.shape
    tensor = tensor.reshape(-1, shape[-1])
    q_matrix, r_matrix = np.linalg.qr(tensor, mode="reduced")
    node.tensor = q_matrix.reshape(shape[:-1] + (q_matrix.shape[1],))
    return r_matrix


def orthonormalize_svd(
    node: TNode, i: int, tol: float, d_max: int, factor: Optional[Tensor] = None
) -> Tensor:
    """Orthonormalization using singular value decomposition.

    After gates are applied, the tree is no longer in a canonical form.
    To restore the form all affected nodes are normalized bottom up.
    The current tensor of a node is split T = U@S@V^T where U is already an isometry.
    The singular values S are multiplied into V and pushed up in the tree.
    Also they can be truncated if necessary.

    Parameters
    ----------
    node: TNode
        The node to normalize.
    i: int
        Index on which the normalization comes from.
    tol: float
        Truncation of small singular values.
    d_max: int
        Maximal allowed dimensionality of connections.
    factor: Optional[Tensor], default = None
        Non-normal part from previous node.

    Returns
    -------
    Tensor
        The non-normal part of the current node.
    """
    tensor = node.tensor
    if factor is not None:  # absorb the result from lower level normalization
        timing = time.perf_counter()
        tensor = contract_factor_on_index(tensor, factor, node.leaf_indices[i])
        logging.debug(
            "Factor %s contraction into %s took %f seconds",
            factor.shape,
            node.tensor.shape,
            time.perf_counter() - timing,
        )

    shape = tensor.shape
    tensor = tensor.reshape(-1, shape[-1])
    u_matrix, singular_values, v_matrix = split_matrix_svd(tensor, tol)
    if len(singular_values) > d_max:
        singular_values = singular_values[:d_max]
        u_matrix = u_matrix[:, :d_max]
        v_matrix = v_matrix[:d_max]
    node.tensor = u_matrix.reshape(shape[:-1] + (u_matrix.shape[1],))
    return np.einsum("a, ab -> ab", singular_values, v_matrix)


def contract_factor_on_index(tensor: Tensor, factor: Tensor, idx: int) -> Tensor:
    """
    Contracts the factor (a NxN matrix) on the given index into the tensor.

    Parameters
    ----------
    tensor: Tensor
        The node tensor.
    factor: Tensor
        Non-normal part to contract.
    idx: int
        Index on which the factor should be contracted.

    Returns
    -------
    Tensor
        The updated tensor.
    """
    params = [tensor, range(0, 2 * tensor.ndim, 2), factor, (2 * idx + 1, 2 * idx)]
    return oe.contract(*params)


def precontract_root(node: TNode, site_j: int, factor: Tensor) -> None:
    """
    Precontracts factors of orthonormalization from two different sites.

    On higher root dimensions (i.e. d=13) wiring the root node becomes
    intractable. Instead the identity gets contracted when the normalization
    remainders arrive from different roots, for example:

    Instead of (2,...,2,...) -> (8,...,8,...) by kronecker product
    keep (2,...,2,...) and add a (4,4) identity temporarily.
    When the normalization remainders arrive contract them along the identity
    and the root.

    Parameters
    ----------
    node: TNode
        The root node with an active tmp_factor.
    site_j: int
        The second site of the gate.
    factor: Tensor
        The normalization remainder of the subtree of site_j.
    """
    assert node.is_root
    tensor = node.tensor
    site_i = node.leaf_indices[node.tmp_index]
    site_j = node.leaf_indices[site_j]

    timing = time.perf_counter()
    node.tensor = oe.contract(
        tensor,
        range(1, tensor.ndim * 2 + 1, 2),
        node.tmp_factor.reshape(node.tmp_factor.shape[0], node.tmp_dim, -1),
        [2 * site_i, tensor.ndim * 2 + 2, 2 * site_i + 1],
        factor.reshape(factor.shape[0], node.tmp_dim, -1),  # remove gate dim
        [2 * site_j, tensor.ndim * 2 + 3, 2 * site_j + 1],
        np.eye(node.tmp_dim),
        [tensor.ndim * 2 + 2, tensor.ndim * 2 + 3],
    )
    logging.debug(
        "Root contraction took %f seconds",
        time.perf_counter() - timing,
    )
    # reset root tmp params
    node.tmp_dim = 0
    node.tmp_factor = None
    node.tmp_index = -1


def pseudo_orthonormalize(
    node: PseudoTNode, i: int, factor: Optional[int] = None
) -> int:
    """
    Pseudo-orthonormalization by correctly updating the shape.

    Parameters
    ----------
    node: PseudoTNode
        The node to pesudo normalize.
    i: int
        Index on which the normalization comes from.
    factor: Optional[int], default = None
        Shape of supposed non-normal part from previous node.

    Returns
    -------
    int
        The theoretical size of the remainder.
    """
    shape = node.shape
    if factor is not None:
        i = node.leaf_indices[i]
        shape = shape[:i] + (factor,) + shape[i + 1 :]
    if float(np.product(tuple(x for x in shape[:-1]), dtype=np.float128)) > shape[-1]:
        node.shape = shape
        return node.shape[-1]
    if not node.is_root:
        node.shape = shape[:-1] + (int(np.product(shape[:-1])),)
    else:
        node.shape = shape
    return node.shape[-1]
