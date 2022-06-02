from typing import NamedTuple, Tuple
from unittest.mock import patch, Mock
import unittest

from sklearn.cluster import SpectralClustering
import numpy as np

from circuits.gates import H, CNOT
from common.circuit import Circuit, CircuitGate
from ttn_circuit_sim.structure import SNode
from ttn_circuit_sim.structure.find_tree_structure import (
    find_tree_structure,
    _to_similarity_matrix,
    _create_subtree,
    _max_cluster_size,
    _create_subtree_flat,
)


class Clustering(NamedTuple):
    labels_: Tuple[int]


class TestFindTreeStructure(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_find_tree_structure(self) -> None:
        circ = Circuit(3, 2)
        circ.append_gate(CircuitGate(H, [0]))
        circ.append_gate(CircuitGate(CNOT, [0, 1], 4))
        circ.append_gate(CircuitGate(CNOT, [0, 2], 2))
        target = SNode(
            "root",
            children=[
                SNode(
                    "0.0",
                    children=[SNode("0.1", children=[SNode(0), SNode(1)]), SNode(2)],
                )
            ],
        )
        self.assertTrue(_nodeEqual(find_tree_structure(circ, random_state=0), target))
        target = SNode(
            "root", children=[SNode("0.0", children=[SNode(0), SNode(1), SNode(2)])]
        )
        self.assertTrue(
            _nodeEqual(find_tree_structure(circ, random_state=0, flat=True), target)
        )
        target = SNode(
            "root",
            children=[SNode("0.0", children=[SNode(0), SNode(1)]), SNode(2)],
        )
        self.assertTrue(
            _nodeEqual(find_tree_structure(circ, clusters=2, find_state=True), target)
        )

    @patch.object(SpectralClustering, "fit")
    def test_find_tree_structure_case(self, mock: Mock) -> None:
        mock.side_effect = lambda x: Clustering(np.array([2, 0, 1]))
        circ = Circuit(3, 2)
        circ.append_gate(CircuitGate(H, [0]))
        circ.append_gate(CircuitGate(CNOT, [0, 1], 4))
        circ.append_gate(CircuitGate(CNOT, [0, 2], 2))
        target = SNode(
            "root",
            children=[SNode(1), SNode(2), SNode(0)],
        )
        self.assertTrue(_nodeEqual(find_tree_structure(circ, clusters=3), target))

    def test_to_similarity_matrix(self) -> None:
        circ = Circuit(3, 2)
        target = np.array([[2 ** 63, 0, 0], [0, 2 ** 63, 0], [0, 0, 2 ** 63]])
        self.assertTrue(np.all(_to_similarity_matrix(circ) == target))
        circ.append_gate(CircuitGate(H, [0]))
        circ.append_gate(CircuitGate(CNOT, [0, 1], 4))
        circ.append_gate(CircuitGate(CNOT, [0, 2], 2))
        target = np.array(
            [
                [2 ** 63, 4 + 1 / 10, 2 + 1 / 8],
                [4 + 1 / 10, 2 ** 63, 0],
                [2 + 1 / 8, 0, 2 ** 63],
            ]
        )
        self.assertTrue(np.allclose(_to_similarity_matrix(circ), target))

    def test_create_subtree(self) -> None:
        similarity = np.array(
            [
                [1, 3, 0, 3, 2],
                [3, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [3, 0, 0, 1, 0],
                [2, 0, 0, 0, 1],
            ]
        )
        leaves = [0, 1, 3, 4]
        target = SNode(
            "1.0",
            children=[SNode("1.1", children=[SNode(0), SNode(1), SNode(3)]), SNode(4)],
        )
        self.assertTrue(_nodeEqual(_create_subtree(leaves, similarity, 1), target))
        leaves = [1]
        target = SNode(1)
        self.assertTrue(_nodeEqual(_create_subtree(leaves, similarity, 1), target))
        leaves = []
        self.assertIsNone(_create_subtree(leaves, similarity, 1))

    def test_create_subtree_flat(self) -> None:
        leaves = [1, 3]
        target = SNode("1.0", children=[SNode(1), SNode(3)])
        self.assertTrue(_nodeEqual(_create_subtree_flat(leaves, 1), target))
        leaves = [1]
        target = SNode(1)
        self.assertTrue(_nodeEqual(_create_subtree_flat(leaves, 1), target))
        leaves = []
        self.assertIsNone(_create_subtree_flat(leaves, 1))

    def test_max_cluster_size(self) -> None:
        labels = list(range(10))
        self.assertEqual(_max_cluster_size(labels), 1)
        labels = [0] * 2 + [1] * 3
        self.assertEqual(_max_cluster_size(labels), 3)


def _nodeEqual(first: SNode, second: SNode) -> bool:
    if not first.is_root:
        if second.is_root:
            return False
        if first.parent.name != second.parent.name:
            return False
    return first.name == second.name and all(
        _nodeEqual(s, o) for s, o in zip(first.children, second.children)
    )
