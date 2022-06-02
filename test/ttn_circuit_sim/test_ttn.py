import unittest

import numpy as np

from circuits.gates import H, CNOT
from common.circuit import Circuit, CircuitGate
from ttn_circuit_sim import apply_circuit, TNode
from ttn_circuit_sim.structure import SNode
from ttn_circuit_sim.tnode import PseudoTNode
from ttn_circuit_sim.ttn import (
    TTN,
    _is_square_identity,
    _find_best_structure,
    _Structure,
)
from .structure.test_single_states_to_tree import _nodeEqual, _pseudoEqual


class TestTTN(unittest.TestCase):
    def setUp(self) -> None:
        self.circ = Circuit(3, 2)
        self.circ.append_gate(CircuitGate(H, [0]))
        self.circ.append_gate(CircuitGate(CNOT, [0, 1], 2))
        self.circ.append_gate(CircuitGate(CNOT, [0, 2], 2))

    def test_basis_state(self) -> None:
        structure = SNode("root", children=[SNode(0), SNode(2), SNode(1)])
        psi = TTN.basis_state(
            2, [0, 0, 0], structure=structure, circ=self.circ, dry_run=True
        )
        apply_circuit(psi, self.circ)
        target = PseudoTNode(
            "root",
            (2, 2, 2, 1),
            children=[
                PseudoTNode(0, (2, 2)),
                PseudoTNode(2, (2, 2)),
                PseudoTNode(1, (2, 2)),
            ],
            leaf_indices={0: 0, 1: 2, 2: 1},
        )
        target.local_dim = 1
        self.assertTrue(_pseudoEqual(psi.root, target))
        psi = TTN.basis_state(2, [0, 0, 0], circ=self.circ, enable_gpu=True, d_max=1)
        apply_circuit(psi, self.circ, compress=True)
        target = TNode(
            "root",
            np.array([[[1]]]),
            children=[
                TNode(
                    "0.0",
                    np.array([[[1]]]),
                    children=[
                        TNode(0, np.array([[1, 0]]).T),
                        TNode(1, np.array([[1, 0]]).T),
                    ],
                    leaf_indices={0: 0, 1: 1},
                ),
                TNode(2, np.array([[1, 0]]).T),
            ],
            leaf_indices={0: 0, 1: 0, 2: 1},
        )
        self.assertAlmostEqual(psi.nrm, 1 / np.sqrt(2))
        self.assertTrue(_nodeEqual(psi.root, target))

    def test_as_vector(self) -> None:
        psi = TTN.basis_state(2, [0, 0, 0], circ=self.circ)
        self.assertTrue(
            np.allclose(psi.as_vector(), np.array([1, 0, 0, 0, 0, 0, 0, 0]))
        )
        apply_circuit(psi, self.circ, compress=True)
        self.assertTrue(
            np.allclose(
                psi.as_vector(), np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
            )
        )

    def test_bond_data(self) -> None:
        psi = TTN.basis_state(2, [0, 0, 0], circ=self.circ)
        self.assertTupleEqual(psi.bond_data(), (8.0, 2))
        apply_circuit(psi, self.circ, compress=True)
        self.assertTupleEqual(psi.bond_data(), (24.0, 2))

    def test_max_leaves(self) -> None:
        psi = TTN.basis_state(2, [0, 0, 0], circ=self.circ)
        self.assertEqual(psi.max_leaves(), 2)
        psi = TTN.basis_state(
            2,
            [0, 0, 0],
            structure=SNode("root", children=[SNode(0), SNode(1), SNode(2)]),
        )
        self.assertEqual(psi.max_leaves(), 1)

    def test_orthonormalize(self) -> None:
        target = np.array([[1, 0], [0, -1]]) / np.sqrt(np.sqrt(2))
        root = TNode(
            "root",
            np.array([[[1]]]),
            children=[TNode(0, target), TNode(1, target)],
            leaf_indices={0: 0, 1: 1},
        )
        root.tmp_dim = 2
        root.tmp_index = 0
        psi = TTN(2, root, enable_gpu=False)
        psi.orthonormalize(0, 1, compress=True)
        target = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.allclose(psi.root[0].tensor, target))
        self.assertTrue(np.allclose(psi.root[1].tensor, target))
        target = np.array([[[1], [0]], [[0], [1]]]) / np.sqrt(2)
        self.assertTrue(np.allclose(psi.root.tensor * psi.nrm, target))
        self.assertEqual(psi.root.tmp_dim, 0)
        self.assertEqual(psi.root.tmp_index, -1)
        self.assertIsNone(psi.root.tmp_factor)


class TestTTNHelpers(unittest.TestCase):
    def test_is_square_identity(self) -> None:
        self.assertTrue(_is_square_identity(np.eye(4)))
        self.assertFalse(_is_square_identity(np.ones(4)))
        self.assertFalse(_is_square_identity(None))
        self.assertFalse(_is_square_identity(2))
        self.assertFalse(_is_square_identity(np.arange(8).reshape(2, 4)))

    def test_find_best_structure(self) -> None:
        structures = {
            0: _Structure(2, 4, 2, 1),
            1: _Structure(3, 4, 3, 2),
            2: _Structure(2, 2, 2, 3),
        }
        self.assertEqual(_find_best_structure(structures), 1)
        self.assertEqual(_find_best_structure(structures, maximize_for_prod=True), 2)
        self.assertEqual(_find_best_structure(structures, bound=2), 2)
        structures.pop(2)
        self.assertEqual(_find_best_structure(structures, bound=2), 1)
