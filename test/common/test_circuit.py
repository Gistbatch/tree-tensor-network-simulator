import unittest

import numpy as np

from common.circuit import CircuitGate, Circuit


class TestCircuit(unittest.TestCase):
    def test_gate(self) -> None:
        cnot = np.identity(4)[[0, 1, 3, 2]]
        gate = CircuitGate(cnot, [0, 1], 2)
        self.assertEqual(gate.dim, 2)
        gate = CircuitGate(cnot, [0, 1])
        self.assertEqual(gate.dim, 4)

    def test_circuit(self) -> None:
        circ = Circuit(2, 2)
        self.assertListEqual(circ.gates, [])
        cnot = np.identity(4)[[0, 1, 3, 2]]
        gate = CircuitGate(cnot, [0, 1], 2)
        circ.append_gate(gate)
        self.assertListEqual(circ.gates, [gate])
