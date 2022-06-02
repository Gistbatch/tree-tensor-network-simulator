"""Impelements the MPS base class."""
from typing import Tuple, NamedTuple
import numpy as np

from common.util import split_matrix_svd, crandn, dim_product
from ttn_circuit_sim.ops.orthonormalize import pseudo_orthonormalize


class PseudoTensor(NamedTuple):
    shape: Tuple[int, int, int]


class MPS:
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(self, d, D, fill="zero", dry_run: bool = False):
        """
        Create a matrix product state.
        """
        self.d = d
        # leading and trailing bond dimensions must agree (typically 1)
        assert D[0] == D[-1]
        if dry_run:
            self.A = [PseudoTensor((d, 1, 1)) for _ in range(len(D) - 1)]
        elif fill == "zero":
            self.A = [np.zeros((d, D[i], D[i + 1])) for i in range(len(D) - 1)]
        elif fill == "random real":
            # random real entries
            self.A = [
                np.random.normal(size=(d, D[i], D[i + 1]))
                / np.sqrt(d * D[i] * D[i + 1])
                for i in range(len(D) - 1)
            ]
        elif fill == "random complex":
            # random complex entries
            self.A = [
                crandn(shape=(d, D[i], D[i + 1])) / np.sqrt(d * D[i] * D[i + 1])
                for i in range(len(D) - 1)
            ]
        else:
            raise ValueError(f"fill = {fill} invalid.")

    def bond_data(self) -> Tuple[float, int]:
        count, current_max = 0, 0
        for entry in self.A:
            count += dim_product(entry.shape)
            current_max = int(max(entry.shape + (current_max,)))
        return count, current_max

    @classmethod
    def basis_state(cls, d, single_states, dry_run: bool = False):
        """
        Represent a computational basis state as MPS;
        `single_states` contains the individual basis states for each site as list.
        """
        L = len(single_states)
        mps = cls(d, (L + 1) * [1], fill="zero", dry_run=dry_run)
        if dry_run:
            return mps
        for i in range(L):
            assert 0 <= single_states[i] < d
            mps.A[i][single_states[i], 0, 0] = 1
        return mps

    @property
    def local_dim(self):
        """
        Local (physical) dimension at each lattice site.
        """
        return self.d

    @property
    def nsites(self):
        """
        Number of lattice sites.
        """
        return len(self.A)

    @property
    def bond_dims(self):
        """
        Virtual bond dimensions.
        """
        if len(self.A) == 0:
            return []
        D = [self.A[i].shape[1] for i in range(len(self.A))]
        D.append(self.A[-1].shape[2])
        return D

    @property
    def dtype(self):
        """
        Data type of tensor entries.
        """
        return self.A[0].dtype

    def orthonormalize(
        self, mode="left", compress=False, tol=0.0, dry_run: bool = False
    ):
        """
        Left- or right-orthonormalize the MPS using SVDs or QR decompositions.
        """
        if len(self.A) == 0:
            return 1

        if dry_run:
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i + 1] = pseudo_orthonormalize(
                    self.A[i], self.A[i + 1]
                )
            self.A[-1], _ = pseudo_orthonormalize(self.A[-1], np.array([[[1.0]]]))
            return 1
        elif compress:
            local_orthonormalize_left = lambda A, Anext: local_orthonormalize_left_svd(
                A, Anext, tol
            )
            local_orthonormalize_right = (
                lambda A, Aprev: local_orthonormalize_right_svd(A, Aprev, tol)
            )
        else:
            local_orthonormalize_left = lambda A, Anext: local_orthonormalize_left_qr(
                A, Anext
            )
            local_orthonormalize_right = lambda A, Aprev: local_orthonormalize_right_qr(
                A, Aprev
            )

        if mode == "left":
            for i in range(len(self.A) - 1):
                self.A[i], self.A[i + 1] = local_orthonormalize_left(
                    self.A[i], self.A[i + 1]
                )
            # last tensor
            self.A[-1], T = local_orthonormalize_left(self.A[-1], np.array([[[1.0]]]))
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[-1] = -self.A[-1]
                nrm = -nrm
            return nrm
        if mode == "right":
            for i in reversed(range(1, len(self.A))):
                self.A[i], self.A[i - 1] = local_orthonormalize_right(
                    self.A[i], self.A[i - 1]
                )
            # first tensor
            self.A[0], T = local_orthonormalize_right(self.A[0], np.array([[[1.0]]]))
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                self.A[0] = -self.A[0]
                nrm = -nrm
            return nrm
        raise ValueError("mode = {} invalid; must be 'left' or 'right'.".format(mode))

    def as_vector(self):
        """
        Merge all tensors to obtain the vector representation on the full Hilbert space.
        """
        psi = self.A[0]
        for i in range(1, len(self.A)):
            psi = merge_mps_tensor_pair(psi, self.A[i])
        assert psi.ndim == 3
        # contract leftmost and rightmost virtual bond
        # has no influence if these virtual bond dimensions are 1
        psi = np.trace(psi, axis1=1, axis2=2)
        return psi


def local_orthonormalize_left_qr(A, Anext):
    """
    Left-orthonormalize a MPS tensor `A` by a QR decomposition,
    and update tensor at next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0] * s[1], s[2])), mode="reduced")
    A = np.reshape(Q, (s[0], s[1], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.transpose(np.tensordot(R, Anext, (1, 1)), (1, 0, 2))
    return A, Anext


def local_orthonormalize_right_qr(A, Aprev):
    """
    Right-orthonormalize a MPS tensor `A` by a QR decomposition,
    and update tensor at previous site.
    """
    # flip left and right virtual bond dimensions
    A = np.transpose(A, (0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0] * s[1], s[2])), mode="reduced")
    A = np.transpose(np.reshape(Q, (s[0], s[1], Q.shape[1])), (0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return A, Aprev


def local_orthonormalize_left_svd(A, Anext, tol):
    """
    Left-orthonormalize a MPS tensor `A` by a (truncated) SVD,
    and update tensor at next site.
    """
    # perform SVD and replace A by reshaped U matrix
    s = A.shape
    assert len(s) == 3
    U, S, V = split_matrix_svd(np.reshape(A, (s[0] * s[1], s[2])), tol)
    A = np.reshape(U, (s[0], s[1], U.shape[1]))
    # update Anext tensor: multiply with diag(S) @ V from left
    Anext = np.transpose(np.tensordot(S[:, None] * V, Anext, (1, 1)), (1, 0, 2))
    return A, Anext


def local_orthonormalize_right_svd(A, Aprev, tol):
    """
    Right-orthonormalize a MPS tensor `A` by a (truncated) SVD,
    and update tensor at previous site.
    """
    # flip left and right virtual bond dimensions
    A = np.transpose(A, (0, 2, 1))
    # perform SVD and replace A by reshaped U matrix
    s = A.shape
    assert len(s) == 3
    U, S, V = split_matrix_svd(np.reshape(A, (s[0] * s[1], s[2])), tol)
    A = np.transpose(np.reshape(U, (s[0], s[1], U.shape[1])), (0, 2, 1))
    # update Aprev tensor: multiply with diag(S) @ V from right
    Aprev = np.tensordot(Aprev, S[:, None] * V, (2, 1))
    return A, Aprev


def merge_mps_tensor_pair(A0, A1):
    """
    Merge two neighboring MPS tensors.
    """
    A = np.tensordot(A0, A1, (2, 1))
    # pair original physical dimensions of A0 and A1
    A = A.transpose((0, 2, 1, 3))
    # combine original physical dimensions
    A = A.reshape((A.shape[0] * A.shape[1], A.shape[2], A.shape[3]))
    return A


def pseudo_orthonormalize(
    a_curr: PseudoTensor, a_next: PseudoTensor
) -> Tuple[PseudoTensor]:
    s_shape = min(a_curr.shape[0] * a_curr.shape[1], a_curr.shape[2])
    a_curr = PseudoTensor((a_curr.shape[0], a_curr.shape[1], s_shape))
    a_next = PseudoTensor((a_next.shape[0], s_shape, a_next.shape[2]))
    return a_curr, a_next
