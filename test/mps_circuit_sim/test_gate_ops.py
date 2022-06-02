import unittest

import numpy as np

import common
import mps_circuit_sim as mcs


class TestGateOps(unittest.TestCase):
    def test_decompose_gate(self):
        """
        Test quantum gate decomposition.
        """
        d = 3
        # random unitary
        U = common.random_unitary(d ** 2)

        V, S, W = mcs.decompose_two_particle_gate(U, d)

        self.assertTrue(
            np.allclose(mcs.reassemble_two_particle_gate(V, S, W), U, atol=1e-10)
        )

    def test_apply_gate(self):
        """
        Test application of unitary gates.
        """
        # create random matrix product state
        d = 3
        D = [1, 4, 7, 13, 10, 5, 1]
        L = len(D) - 1
        mps = mcs.MPS(d, D, fill="random complex")
        psi = mps.as_vector()

        # random single- and two-particle unitaries
        U1 = common.random_unitary(d)
        U2 = common.random_unitary(d ** 2)
        U3 = common.random_unitary(d ** 2)

        # `mps` is overwritten in-place
        mcs.apply_single_particle_gate(mps, U1, 3)
        mcs.apply_two_particle_gate(mps, U2, 2, 4)
        mcs.apply_two_particle_gate(mps, U3, 1, 4)

        # reference calculation
        Upsi = apply_two_particle_gate_vec(
            apply_two_particle_gate_vec(
                apply_single_particle_gate_vec(psi, U1, 3, L, d), U2, 2, 4, L, d
            ),
            U3,
            1,
            4,
            L,
            d,
        )

        self.assertTrue(np.allclose(mps.as_vector(), Upsi, atol=1e-10))


def apply_single_particle_gate_vec(psi: np.ndarray, U, i, L, d=2):
    """
    Apply the single-particle gate `U` acting on site `i`
    to the quantum state `psi` represented as vector.
    """
    assert 0 <= i < L
    psi = np.reshape(psi, (d ** i, d, d ** (L - i - 1)))
    psi = np.einsum(U, (1, 2), psi, (0, 2, 3), (0, 1, 3))
    psi = np.reshape(psi, -1)
    return psi


def apply_two_particle_gate_vec(psi: np.ndarray, U, i, j, L, d=2):
    """
    Apply the two-particle gate `U` acting on sites `i` and `j`
    to the quantum state `psi` represented as vector.
    """
    assert 0 <= i < j < L
    U = np.reshape(U, (d, d, d, d))
    psi = np.reshape(psi, (d ** i, d, d ** (j - i - 1), d, d ** (L - j - 1)))
    psi = np.einsum(U, (1, 5, 2, 4), psi, (0, 2, 3, 4, 6), (0, 1, 3, 5, 6))
    psi = np.reshape(psi, -1)
    return psi


if __name__ == "__main__":
    unittest.main()
