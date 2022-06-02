import unittest

import numpy as np

import mps_circuit_sim as mcs


class TestMPS(unittest.TestCase):
    def test_basis_state(self):
        """
        Test MPS representation of a computational basis state.
        """
        d = 5
        states = [1, 0, 3, 4, 2, 0, 1]
        mps = mcs.MPS.basis_state(d, states)
        psi = np.array([1.0])
        for i in states:
            phi = np.zeros(d)
            phi[i] = 1
            psi = np.kron(psi, phi)
        self.assertTrue(
            np.allclose(mps.as_vector(), psi, atol=1e-10),
            msg="computational basis state represented as MPS must match reference",
        )

    def test_orthonormalization(self):
        """
        Test MPS orthonormalization.
        """
        # create random matrix product state
        d = 7
        D = [1, 4, 15, 13, 7, 1]
        mps = mcs.MPS(d, D, fill="random complex")

        self.assertEqual(mps.bond_dims, D, msg="virtual bond dimensions")

        # wavefunction on full Hilbert space
        psi = mps.as_vector()

        # performing left-orthonormalization...
        cL = mps.orthonormalize(mode="left", compress=True, tol=1e-8)

        self.assertLessEqual(
            mps.bond_dims[1],
            d,
            msg="virtual bond dimension can only increase by a factor of 'd' per site",
        )

        psiL = mps.as_vector()
        # wavefunction should now be normalized
        self.assertAlmostEqual(
            np.linalg.norm(psiL), 1.0, delta=1e-12, msg="wavefunction normalization"
        )

        # wavefunctions before and after left-normalization must match
        # (up to normalization factor)
        self.assertTrue(
            np.allclose(cL * psiL, psi, atol=1e-10),
            msg="wavefunctions before and after left-normalization must match",
        )

        # check left-orthonormalization
        for i in range(mps.nsites):
            s = mps.A[i].shape
            assert s[0] == d
            Q = mps.A[i].reshape((s[0] * s[1], s[2]))
            self.assertTrue(is_isometry(Q), msg="left-orthonormalization")

        # performing right-orthonormalization...
        cR = mps.orthonormalize(mode="right", compress=True, tol=1e-8)

        self.assertLessEqual(
            mps.bond_dims[-2],
            d,
            msg="virtual bond dimension can only increase by a factor of 'd' per site",
        )

        self.assertAlmostEqual(
            abs(cR),
            1.0,
            delta=1e-12,
            msg="normalization factor must have magnitude 1 due to previous left-orthonormalization",
        )

        psiR = mps.as_vector()
        # wavefunctions must match
        self.assertTrue(
            np.allclose(psiL, cR * psiR, atol=1e-10),
            msg="wavefunctions after left- and right-orthonormalization must match",
        )

        # check right-orthonormalization
        for i in range(mps.nsites):
            s = mps.A[i].shape
            assert s[0] == d
            Q = mps.A[i].transpose((0, 2, 1)).reshape((s[0] * s[2], s[1]))
            self.assertTrue(is_isometry(Q), msg="right-orthonormalization")


def is_isometry(A):
    """
    Test if `A` is an isometry.
    """
    return np.allclose(np.dot(A.conj().T, A), np.identity(A.shape[1]), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
