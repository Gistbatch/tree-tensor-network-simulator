"""Implements the gate operations."""
import numpy as np

from common.circuit import Circuit
from .mps import MPS, PseudoTensor


def apply_single_particle_gate(psi: MPS, U, i, dry_run: bool = False):
    """
    Apply a single-particle quantum gate acting on site `i`
    to a quantum state `psi` represented as MPS. `psi` is updated in-place.
    """
    assert 0 <= i < psi.nsites
    if dry_run:
        return
    psi.A[i] = np.tensordot(U, psi.A[i], axes=1)


def decompose_two_particle_gate(U, d=2):
    """
    Decompose a two-particle quantum gate using the singular value decomposition.
    """
    U = np.reshape(U, (d, d, d, d))
    U = np.transpose(U, (0, 2, 1, 3))
    U = np.reshape(U, (d ** 2, d ** 2))
    V, S, W = np.linalg.svd(U)
    W = W.T
    idx = S > 1e-14  # select the analytically non-zero singular values
    V = V[:, idx]
    W = W[:, idx]
    S = S[idx]
    V = np.reshape(V, (d, d, -1))
    W = np.reshape(W, (d, d, -1))
    return V, S, W


def reassemble_two_particle_gate(V, S, W):
    """
    Reassemble a two-particle quantum gate from its decomposed version.
    """
    return np.sum(
        [S[j] * np.kron(V[:, :, j], W[:, :, j]) for j in range(len(S))], axis=0
    )


def apply_two_particle_gate(psi: MPS, U, i, j, dry_run: bool = False):
    """
    Apply a two-particle quantum gate acting on sites `i` and `j`
    to a quantum state `psi` represented as MPS. `psi` is updated in-place.
    """
    assert 0 <= i < j < psi.nsites
    V, S, W = decompose_two_particle_gate(U, psi.local_dim)
    # absorb singular values in V and W
    V *= np.sqrt(S)
    W *= np.sqrt(S)
    if dry_run:
        dim = V.shape[2]
        shape = psi.A[i].shape
        psi.A[i] = PseudoTensor((shape[0], shape[1], shape[2] * dim))
        for k in range(i + 1, j):
            shape = psi.A[k].shape
            psi.A[k] = PseudoTensor((shape[0], shape[1] * dim, shape[2] * dim))
        shape = psi.A[j].shape
        psi.A[j] = PseudoTensor((shape[0], shape[1] * dim, shape[2]))
        return
    # site `i`
    psi.A[i] = np.einsum(V, (0, 1, 2), psi.A[i], (1, 3, 4), (0, 3, 2, 4))
    sp = psi.A[i].shape
    psi.A[i] = np.reshape(psi.A[i], (sp[0], sp[1], sp[2] * sp[3]))
    # thread gate bond through MPS
    for k in range(i + 1, j):
        psi.A[k] = np.array(
            [
                np.kron(np.identity(len(S)), psi.A[k][n, :, :])
                for n in range(psi.local_dim)
            ]
        )
    # site `j`
    psi.A[j] = np.einsum(W, (0, 1, 2), psi.A[j], (1, 3, 4), (0, 2, 3, 4))
    sp = psi.A[j].shape
    psi.A[j] = np.reshape(psi.A[j], (sp[0], sp[1] * sp[2], sp[3]))


def apply_circuit(
    psi: MPS, circ: Circuit, compress=False, tol=0.0, dry_run: bool = False
):
    """
    Apply the gates of the quantum circuit `circ` to the quantum state `psi`
    represented as MPS. `psi` is updated in-place.
    """
    # dimension compatibility checks
    assert psi.local_dim == circ.local_dimension
    assert psi.nsites == circ.l_sites
    for g in circ.gates:
        assert 1 <= len(g.sites) <= 2, "only single- and two-particle gates supported"
        if len(g.sites) == 1:
            apply_single_particle_gate(psi, g.gate_matrix, g.sites[0], dry_run)
        else:
            apply_two_particle_gate(psi, g.gate_matrix, g.sites[0], g.sites[1], dry_run)
            psi.orthonormalize("left", compress, tol, dry_run)
