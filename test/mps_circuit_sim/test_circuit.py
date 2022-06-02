import unittest

import numpy as np

import common
import mps_circuit_sim as mcs


class TestCircuit(unittest.TestCase):
    def test_apply_circuit(self):
        """
        Test application of a quantum circuit to a quantum state.
        """
        # create random matrix product state
        d = 3
        D = [1, 3, 10, 8, 12, 7, 1]
        L = len(D) - 1
        mps = mcs.MPS(d, D, fill="random complex")
        mps.orthonormalize("left", compress=True)
        psi = mps.as_vector()

        circ = common.Circuit(L, d)
        ngates = 12
        for n in range(ngates):
            if np.random.randint(2) == 0:
                # single-particle gate
                circ.append_gate(
                    common.CircuitGate(common.random_unitary(d), [np.random.randint(L)])
                )
            else:
                # two-particle gate
                ij = list(np.sort(np.random.choice(L, size=2, replace=False)))
                circ.append_gate(common.CircuitGate(common.random_unitary(d ** 2), ij))

        # `mps` is overwritten in-place
        mcs.apply_circuit(mps, circ, compress=True, tol=1e-10)

        # reference calculation
        for g in circ.gates:
            if len(g.sites) == 1:
                psi = apply_single_particle_gate_vec(
                    psi, g.gate_matrix, g.sites[0], L, d
                )
            else:
                psi = apply_two_particle_gate_vec(
                    psi, g.gate_matrix, g.sites[0], g.sites[1], L, d
                )

        self.assertTrue(np.allclose(mps.as_vector(), psi, atol=1e-10))


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
