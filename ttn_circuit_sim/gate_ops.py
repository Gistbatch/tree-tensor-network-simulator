"""Implements procedures for gate application."""
from typing import Tuple
import logging
import time

import numpy as np

from common import Circuit, Tensor


def _apply_single_particle_gate(psi, gate_matrix: Tensor, site: int) -> None:
    """Applies a single particle gate to the tree tensor.

    This is a simple matrix multiplication with the current leaf tensor.

    Parameters
    ----------
    psi: TTN
        The tree tensor wrapper.
    gate_matrix: Tensor
        The (2,2) unitary gate matrix.
    site: int
        The affected qubit.
    """
    assert 0 <= site < psi.nsites
    psi.root[site].apply_gate(gate_matrix)


def _decompose_two_particle_gate(
    gate_matrix: Tensor, local_dimension: int = 2
) -> Tuple[Tensor]:
    """Decompose a two-particle quantum gate using the singular value decomposition.

    G = U @ S @ V^T. Small singular values are automatically truncated.
    The gate has to be split along the corret axis.

    Parameters
    ----------
    gate_matrix: Tensor
        The (4,4) unitary gate matrix.
    local_dimension: int, default = 2
        The qubit dimension.

    Returns
    -------
    Tuple[Tensor]
        The split matrices U @ S @ V^T = A.
    """
    gate_matrix = np.reshape(
        gate_matrix,
        (local_dimension, local_dimension, local_dimension, local_dimension),
    )
    gate_matrix = np.transpose(
        gate_matrix, (0, 2, 1, 3)
    )  # switch to the correct axis split
    # svd only works matrices (n,m)
    gate_matrix = np.reshape(gate_matrix, (local_dimension ** 2, local_dimension ** 2))
    u_matrix, singular_values, v_matrix = np.linalg.svd(gate_matrix)
    v_matrix = v_matrix.T
    idx = singular_values > 1e-14  # select the analytically non-zero singular values
    u_matrix = u_matrix[:, idx]
    v_matrix = v_matrix[:, idx]
    singular_values = singular_values[idx]
    # return the gate as 3-tensor (physical, tree, gate) so gate can be threaded
    u_matrix = np.reshape(u_matrix, (local_dimension, local_dimension, -1))
    v_matrix = np.reshape(v_matrix, (local_dimension, local_dimension, -1))
    return u_matrix, singular_values, v_matrix


def _apply_two_particle_gate(
    psi, gate_matrix: Tensor, site_i: int, site_j: int
) -> None:
    """Applies a two particle gate to the tree tensor.

    First split the gate with svd and update leaves, with U and V respectively.
    Then wire the resulting dimensions through the tree by increasing dimensions
    of all intermediate nodes.

    Parameters
    ----------
    psi: TTN
        The tree tensor wrapper.
    gate_matrix: Tensor
        The (4,4) unitary gate matrix.
    site_i: int
        The first affected qubit.
    site_j: int
        The second affected qubit.
    """
    assert 0 <= site_i < site_j < psi.nsites
    u_matrix, singular_values, v_matrix = _decompose_two_particle_gate(
        gate_matrix, psi.local_dim
    )
    u_matrix *= np.sqrt(singular_values)
    v_matrix *= np.sqrt(singular_values)
    # update leaves
    psi.root[site_i].apply_gate_and_reshape(u_matrix)
    psi.root[site_j].apply_gate_and_reshape(v_matrix)
    gate_dim = len(singular_values)
    # wire all common nodes on the path from i to j
    # i and j are already handled (leaves)
    for node in psi.root[site_i:site_j][1:-1]:
        timing = time.perf_counter()
        node.update(gate_dim, site_i, site_j)
        timing = time.perf_counter() - timing
        if timing > 10:
            logging.warning(
                "Wiring of %s with shape %s took %f seconds",
                node.name,
                node.tensor.shape,
                timing,
            )
        else:
            logging.debug(
                "Wiring of %s with shape %s took %f seconds",
                node.name,
                node.tensor.shape,
                timing,
            )


def apply_circuit(psi, circ: Circuit, compress: bool = False, tol: float = 0.0) -> None:
    """Applies all gates to the tree tensor.

    Normalization after each two particle gate is necessary.

    Parameters
    ----------
    psi: TTN
        The tree tensor wrapper.
    circ: Circuit
        The circuit definition.
    compress: bool, default = False
        Use compression in normalization.
    tol: float, default = 0.0
        The compression tolerance.
    """
    assert psi.local_dim == circ.local_dimension
    assert psi.nsites == circ.l_sites
    for idx, gate in enumerate(circ.gates):
        logging.debug("Gate %d on %s", idx, gate.sites)
        logging.debug("-" * 80)
        timing = time.perf_counter()
        assert (
            1 <= len(gate.sites) <= 2
        ), "only single- and two-particle gates supported"
        if len(gate.sites) == 1:
            _apply_single_particle_gate(psi, gate.gate_matrix, gate.sites[0])
        else:
            _apply_two_particle_gate(
                psi, gate.gate_matrix, gate.sites[0], gate.sites[1]
            )
            psi.orthonormalize(gate.sites[0], gate.sites[1], compress, tol)
        timing = time.perf_counter() - timing
        if timing > 10:
            logging.warning("Gate %d on %s took %f seconds \n", idx, gate.sites, timing)
        else:
            logging.debug("Gate %d on %s took %f seconds \n", idx, gate.sites, timing)
