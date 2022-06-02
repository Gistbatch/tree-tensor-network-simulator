import unittest

import numpy as np

from circuits import X, CNOT
from ttn_circuit_sim.tnode import PseudoTNode, TNode
from .structure.test_single_states_to_tree import _nodeEqual, _pseudoEqual


class TestTNode(unittest.TestCase):
    def test_get_item(self) -> None:
        root = TNode(
            "root",
            np.array([1]),
            children=[
                TNode(1, np.array([1])),
                TNode(
                    "1.0",
                    np.array([1]),
                    children=[
                        TNode(0, np.array([1])),
                        TNode(2, np.array([1])),
                    ],
                ),
            ],
        )
        self.assertTrue(_nodeEqual(root[0], root.children[1].children[0]))
        self.assertTrue(_nodeEqual(root[1], root.children[0]))
        self.assertTrue(_nodeEqual(root[2], root.children[1].children[1]))

        self.assertListEqual([x.name for x in root[0:]], [0, "1.0", "root"])
        self.assertListEqual([x.name for x in root[1:]], [1, "root"])
        self.assertListEqual([x.name for x in root[2:]], [2, "1.0", "root"])

        self.assertListEqual([x.name for x in root[0:1]], [0, "1.0", "root", 1])
        self.assertListEqual([x.name for x in root[0:2]], [0, "1.0", 2])
        self.assertListEqual([x.name for x in root[1:2]], [1, "root", "1.0", 2])

    def test_apply_gate(self) -> None:
        node = TNode(0, np.array([[1, 0]]).T)
        self.assertIsNone(node.apply_gate(X))
        self.assertTrue(np.all(node.tensor == np.array([[0, 1]]).T))

    def test_apply_gate_and_reshape(self) -> None:
        node = TNode(0, np.array([[1, 0]]).T)
        gate = np.array([[[1, 0], [0, 0]], [[0, 0], [0, -1]]]) * np.sqrt(np.sqrt(2))
        self.assertIsNone(node.apply_gate_and_reshape(gate))
        target = np.array([[1, 0], [0, 0]]) * np.sqrt(np.sqrt(2))
        self.assertTrue(np.all(node.tensor == target))

    def test_update(self) -> None:
        root = TNode(
            "root",
            np.array([[[1]]]),
            children=[
                TNode(1, np.array([[1, 0]]).T),
                TNode(
                    "1.0",
                    np.array([[[1]]]),
                    children=[
                        TNode(0, np.array([[1, 0]]).T),
                        TNode(2, np.array([[1, 0]]).T),
                    ],
                    leaf_indices={0: 0, 2: 1},
                ),
            ],
            leaf_indices={0: 1, 1: 0, 2: 1},
        )
        self.assertIsNone(root.children[1].update(4, 0, 1))
        target = np.eye(4).reshape(4, 1, 4)
        self.assertTrue(np.all(root.children[1].tensor == target))
        self.assertIsNone(root.update(4, 0, 1))
        self.assertEqual(root.tmp_dim, 4)
        self.assertEqual(root.tmp_index, 0)


class TestPseudoTNode(unittest.TestCase):
    def test_get_item(self) -> None:
        root = PseudoTNode(
            "root",
            (1,),
            children=[
                PseudoTNode(1, (2, 2)),
                PseudoTNode(
                    "1.0", (1,), children=[PseudoTNode(0, (1, 1)), PseudoTNode(2, (1,))]
                ),
            ],
        )
        self.assertTrue(_pseudoEqual(root[0], root.children[1].children[0]))
        self.assertTrue(_pseudoEqual(root[1], root.children[0]))
        self.assertTrue(_pseudoEqual(root[2], root.children[1].children[1]))

        self.assertListEqual([x.name for x in root[0:]], [0, "1.0", "root"])
        self.assertListEqual([x.name for x in root[1:]], [1, "root"])
        self.assertListEqual([x.name for x in root[2:]], [2, "1.0", "root"])

        self.assertListEqual([x.name for x in root[0:1]], [0, "1.0", "root", 1])
        self.assertListEqual([x.name for x in root[0:2]], [0, "1.0", 2])
        self.assertListEqual([x.name for x in root[1:2]], [1, "root", "1.0", 2])

    def test_apply_gate(self) -> None:
        node = PseudoTNode(0, (2, 1))
        self.assertIsNone(node.apply_gate(X))
        self.assertTupleEqual(node.shape, (2, 1))

    def test_apply_gate_and_reshape(self) -> None:
        node = PseudoTNode(0, (2, 1))
        self.assertIsNone(node.apply_gate_and_reshape(CNOT))
        self.assertTupleEqual(node.shape, (2, 4))

    def test_update(self) -> None:
        root = PseudoTNode(
            "root",
            (1, 1, 1),
            children=[
                PseudoTNode(1, (2, 2)),
                PseudoTNode(
                    "1.0",
                    (1, 1, 1),
                    children=[PseudoTNode(0, (1, 1)), PseudoTNode(2, (1,))],
                    leaf_indices={0: 0, 2: 1},
                ),
            ],
            leaf_indices={0: 1, 1: 0, 2: 1},
        )
        self.assertIsNone(root.children[1].update(4, 0, 1))
        self.assertTupleEqual(root.children[1].shape, (4, 1, 4))
        self.assertIsNone(root.update(4, 0, 1))
        self.assertTupleEqual(root.shape, (4, 4, 1))
