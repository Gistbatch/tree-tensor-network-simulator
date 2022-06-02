import unittest

import numpy as np

from ttn_circuit_sim.tnode import PseudoTNode, TNode
from ttn_circuit_sim.structure import SNode, create_pseudo_tree
from ttn_circuit_sim.structure.single_states_to_tree import _create_tree_level


class TestSingleStatesToTree(unittest.TestCase):
    def test_create_pseudo_tree(self) -> None:
        tree = SNode(
            "root", children=[SNode(0), SNode("0.1", children=[SNode(1), SNode(2)])]
        )
        leaf = SNode(0)
        self.assertTrue(
            _pseudoEqual(create_pseudo_tree(leaf, [0], 2), PseudoTNode(0, (2, 1)))
        )
        target = PseudoTNode(
            "root",
            (1, 1, 1),
            children=[
                PseudoTNode(0, (3, 1)),
                PseudoTNode(
                    "0.1",
                    (1, 1, 1),
                    children=[
                        PseudoTNode(1, (3, 1)),
                        PseudoTNode(2, (3, 1)),
                    ],
                    leaf_indices={1: 0, 2: 1},
                ),
            ],
            leaf_indices={0: 0, 1: 1, 2: 1},
        )
        self.assertTrue(_pseudoEqual(create_pseudo_tree(tree, [0, 1, 2], 3), target))

    def test_create_tree_level(self) -> None:
        tree = SNode(
            "root", children=[SNode(0), SNode("0.1", children=[SNode(1), SNode(2)])]
        )
        leaf = SNode(0)
        self.assertTrue(
            _nodeEqual(_create_tree_level(leaf, [0], 2), TNode(0, np.array([[1, 0]]).T))
        )
        target = TNode(
            "root",
            np.array([[[1]]]),
            children=[
                TNode(0, np.array([[1, 0, 0]]).T),
                TNode(
                    "0.1",
                    np.array([[[1]]]),
                    children=[
                        TNode(1, np.array([[0, 1, 0]]).T),
                        TNode(2, np.array([[0, 0, 1]]).T),
                    ],
                    leaf_indices={1: 0, 2: 1},
                ),
            ],
            leaf_indices={0: 0, 1: 1, 2: 1},
        )
        self.assertTrue(_nodeEqual(_create_tree_level(tree, [0, 1, 2], 3), target))


def _pseudoEqual(first: PseudoTNode, second: PseudoTNode) -> bool:
    if not first.is_root:
        if second.is_root:
            return False
        if first.parent.name != second.parent.name:
            return False
    return (
        first.name == second.name
        and first.shape == second.shape
        and first.local_dim == second.local_dim
        and first.leaf_indices == second.leaf_indices
        and first.tmp_dim == second.tmp_dim
        and all(_pseudoEqual(s, o) for s, o in zip(first.children, second.children))
    )


def _nodeEqual(first: TNode, second: TNode) -> bool:
    if not first.is_root:
        if second.is_root:
            return False
        if first.parent.name != second.parent.name:
            return False
    return (
        first.name == second.name
        and np.allclose(first.tensor, second.tensor)
        and first.local_dim == second.local_dim
        and first.leaf_indices == second.leaf_indices
        and first.tmp_dim == second.tmp_dim
        and first.tmp_index == second.tmp_index
        and first.tmp_factor == second.tmp_factor
        and all(_nodeEqual(s, o) for s, o in zip(first.children, second.children))
    )
