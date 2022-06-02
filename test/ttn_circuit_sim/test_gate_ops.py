import unittest

import numpy as np

from circuits.gates import H, CNOT
from common.circuit import Circuit, CircuitGate
from ttn_circuit_sim import TTN
from ttn_circuit_sim.gate_ops import (
    _apply_single_particle_gate,
    _decompose_two_particle_gate,
    _apply_two_particle_gate,
    apply_circuit,
)


class TestGateOps(unittest.TestCase):
    def test_apply_single_particle_gate(self) -> None:
        circ = Circuit(1, 2)
        psi = TTN.basis_state(2, [0], circ=circ)
        _apply_single_particle_gate(psi, H, 0)
        self.assertTrue(np.all(psi.root.tensor == np.array([[1, 1]]).T / np.sqrt(2)))

    def test_decompose_two_particle_gate(self) -> None:
        U, S, V = _decompose_two_particle_gate(CNOT)
        self.assertTrue(
            np.allclose(U, np.array([1, 0, 0, 0, 0, 0, 0, -1]).reshape(2, 2, 2))
        )
        self.assertTrue(np.all(S == np.array([1, 1]) * np.sqrt(2)))
        self.assertTrue(
            np.allclose(
                V, np.array([1, 0, 0, -1, 0, -1, 1, 0]).reshape(2, 2, 2) / np.sqrt(2)
            )
        )

    def test_apply_two_particle_gate(self) -> None:
        circ = Circuit(2, 2)
        psi = TTN.basis_state(2, [0, 0], circ=circ)
        _apply_two_particle_gate(psi, CNOT, 0, 1)
        target = np.array([[1, 0], [0, 0]]) * np.sqrt(np.sqrt(2))
        self.assertTrue(np.allclose(psi.root[0].tensor, target))
        target = np.array([[1, 0], [0, -1]]) / np.sqrt(np.sqrt(2))
        self.assertTrue(np.allclose(psi.root[1].tensor, target))
        self.assertTupleEqual(psi.root.tensor.shape, (1, 1, 1))
        self.assertEqual(psi.root.tmp_dim, 2)
        self.assertEqual(psi.root.tmp_index, 0)

    def test_apply_circuit(self) -> None:
        circ = Circuit(2, 2)
        psi = TTN.basis_state(2, [0, 0], circ=circ)
        circ.append_gate(CircuitGate(H, [0]))
        circ.append_gate(CircuitGate(CNOT, [0, 1], 4))
        apply_circuit(psi, circ)
        target = np.array([[1, 0], [0, 1]])
        self.assertTrue(np.allclose(psi.root[0].tensor, target))
        self.assertTrue(np.allclose(psi.root[1].tensor, target))
        target = np.array([[[1], [0]], [[0], [1]]]) / np.sqrt(2)
        self.assertTrue(np.allclose(psi.root.tensor * psi.nrm, target))
        self.assertEqual(psi.root.tmp_dim, 0)
        self.assertEqual(psi.root.tmp_index, -1)
        self.assertIsNone(psi.root.tmp_factor)
