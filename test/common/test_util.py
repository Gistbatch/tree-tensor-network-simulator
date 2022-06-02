import unittest

import numpy as np

from common.util import (
    random_unitary,
    retained_bond_indices,
    split_matrix_svd,
    dim_product,
)


class TestUtil(unittest.TestCase):
    def test_unitary(self) -> None:
        matrix = random_unitary(4)
        self.assertTrue(np.allclose(matrix @ matrix.conj().T, np.identity(4)))

    def test_retained_bond_indices(self) -> None:
        tensor = np.zeros(16)
        self.assertTrue(all(retained_bond_indices(tensor, 1) == np.array([])))
        tensor = np.ones(16)
        self.assertTrue(all(retained_bond_indices(tensor, 1) == np.array([])))
        self.assertTrue(
            all(retained_bond_indices(tensor, 0.1) == np.array(range(1, 16)))
        )
        self.assertTrue(all(retained_bond_indices(tensor, 0) == np.array(range(16))))

    def test_svd(self) -> None:
        cnot = (
            np.identity(4)[[0, 1, 3, 2]]
            .reshape(2, 2, 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(4, 4)
        )
        U, S, V = split_matrix_svd(cnot, 0)
        self.assertTupleEqual(U.shape, (4, 2))
        self.assertTupleEqual(S.shape, (2,))
        self.assertTupleEqual(V.shape, (2, 4))
        self.assertTrue(np.allclose(cnot, np.einsum("ab, b, bc", U, S, V)))
        U, S, V = split_matrix_svd(cnot, 0.6)
        self.assertTupleEqual(U.shape, (4, 1))
        self.assertTupleEqual(S.shape, (1,))
        self.assertTupleEqual(V.shape, (1, 4))
        cnot[0] = np.zeros(4)
        self.assertTrue(np.allclose(cnot, np.einsum("ab, b, bc", U, S, V)))

    def test_dim_product(self) -> None:
        self.assertEqual(dim_product(range(6)), 120.0)
        self.assertEqual(dim_product([1] * 5), 1.0)
