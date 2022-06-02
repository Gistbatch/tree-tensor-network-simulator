import unittest

import numpy as np

from ttn_circuit_sim.ops import PseudoTNode, TNode
from ttn_circuit_sim.ops.orthonormalize import (
    orthonormalize_qr,
    orthonormalize_svd,
    contract_factor_on_index,
    precontract_root,
    pseudo_orthonormalize,
)


class TestOrthonormalize(unittest.TestCase):
    def test_orthonormalize_qr(self) -> None:
        identity = np.identity(2)
        factor = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        leaves = [TNode(0, identity), TNode(1, factor)]
        node = TNode(
            "root",
            np.array([[[1], [0], [1j], [0j]], [[0], [-1j], [0], [1]]]) / np.sqrt(2),
            leaf_indices={0: 0, 1: 1},
            children=leaves,
        )
        self.assertTrue(np.all(orthonormalize_qr(leaves[1], 0) == factor))
        self.assertTrue(np.all(leaves[1].tensor == identity))
        self.assertTrue(np.all(orthonormalize_qr(node, 1, factor) == np.array([[-1]])))
        self.assertTrue(
            np.all(node.tensor == np.array([[[1], [1j]], [[0], [0]]]) / -np.sqrt(2))
        )

    def test_orthonormalize_svd(self) -> None:
        identity = np.identity(2)
        factor = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        leaves = [TNode(0, identity), TNode(1, factor)]
        node = TNode(
            "root",
            np.array([[[1], [0], [1j], [0j]], [[0], [-1j], [0], [1]]]) / np.sqrt(2),
            leaf_indices={0: 0, 1: 1},
            children=leaves,
        )
        self.assertTrue(np.all(orthonormalize_svd(leaves[1], 0, 0, 8) == factor))
        self.assertTrue(np.all(leaves[1].tensor == identity))
        self.assertTrue(
            np.all(orthonormalize_svd(node, 1, 0, 8, factor) == np.array([[1]]))
        )
        self.assertTrue(
            np.all(node.tensor == np.array([[[1], [1j]], [[0], [0]]]) / np.sqrt(2))
        )

    def test_contract_factor_on_index(self) -> None:
        tensor = np.arange(16).reshape(2, 4, 2)
        factor = np.array([[1, 0], [0, -1]])
        target = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, -8, -9, -10, -11, -12, -13, -14, -15]
        ).reshape(2, 4, 2)
        self.assertTrue(np.all(contract_factor_on_index(tensor, factor, 0) == target))
        target = np.array(
            [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15]
        ).reshape(2, 4, 2)
        self.assertTrue(np.all(contract_factor_on_index(tensor, factor, 2) == target))

    def test_percontract_root(self) -> None:
        factor = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        node = TNode(
            "root",
            np.array([[[[1]]]]),
            leaf_indices={0: 0, 1: 1},
        )
        node.tmp_dim = 4
        node.tmp_factor = np.array([[1, 0, 1j, 0], [0, -1, 0, -1j]])
        node.tmp_index = 0
        self.assertIsNone(precontract_root(node, 1, factor))
        self.assertTrue(
            np.all(node.tensor == np.array([[[[1]], [[1j]]], [[[0]], [[0]]]]))
        )

    def test_pesudo_orthonormalize(self) -> None:
        leaves = [PseudoTNode(0, (2, 1)), PseudoTNode(1, (2, 4))]
        node = PseudoTNode(
            "root", (2, 4, 1), leaf_indices={0: 0, 1: 1}, children=leaves
        )
        self.assertEqual(pseudo_orthonormalize(leaves[1], 0), 2)
        self.assertTupleEqual(leaves[1].shape, (2, 2))
        self.assertEqual(pseudo_orthonormalize(node, 1, 2), 1)
        self.assertTupleEqual(node.shape, (2, 2, 1))
