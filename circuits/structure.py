"""Creates the `Structure` circuit example."""
from typing import Tuple

from qiskit import QuantumCircuit
import numpy as np

from common import Circuit, CircuitGate
from .gates import CNOT, GATE_SET, Q_GATE_SET

D = 2


def apply_single_gates(circ: Circuit, qqc: QuantumCircuit, size: int) -> None:
    """
    Apply the randomized single gates.

    Parameters
    ----------
    circ: Circuit
        MPS/TTN circuit.
    qqc: QuantumCircuit
        Qiskit circuit.
    size: int
        The number of qubits.
    """
    draws = list(np.random.randint(0, 3, size=size))
    for qubit in range(circ.l_sites):
        draw = draws.pop()
        circ.append_gate(CircuitGate(GATE_SET[draw], [qubit]))
        qqc.append(Q_GATE_SET[draw], [qubit])


def create_structure_circuit(units: int = 4) -> Tuple[Circuit, QuantumCircuit]:
    """
    Creates circuits for simulation.

    Parameters
    ----------
    units: int, default = 4
        Defines the number of tightly couppled 4 qubit units.

    Returns
    -------
    Tuple[Circuit, QuantumCircuit]
        The circuits, one for the mps/ttn simulator and one for qiskit.
    """
    sites = (units * 4) + 1
    a_gates = [[x, x + 1] for x in range(0, sites - 2, 4)]
    b_gates = [[x, x + 1] for x in range(1, sites - 2, 4)]
    c_gates = [[x, x + 1] for x in range(2, sites - 2, 4)]
    d_gates = [[x, sites - 1] for x in range(3, sites - 1, 4)]
    pattern = [
        a_gates,
        b_gates,
        c_gates,
        d_gates,
        a_gates,
        b_gates,
        c_gates,
        a_gates,
        b_gates,
        c_gates,
        a_gates,
        b_gates,
    ]
    structure = Circuit(sites, D)
    qqc = QuantumCircuit(sites)
    for step in pattern:
        apply_single_gates(structure, qqc, sites)
        for pair in step:
            structure.append_gate(CircuitGate(CNOT, pair, 2))
            qqc.cx(pair[0], pair[1])
    return structure, qqc
