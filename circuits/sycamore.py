"""Creates the Sycamore circuit example."""
from typing import List, Tuple

from qiskit import QuantumCircuit

from common import Circuit, CircuitGate
from .gates import Q_ROOT_ISWAP, ROOT_ISWAP

from .structure import apply_single_gates


D = 2


def create_sycamore_circuit(rows: int = 4) -> Tuple[Circuit, QuantumCircuit]:
    """
    Creates circuits for simulation.

    Parameters
    ----------
    rows: int, default = 4
        Defines the number of rows in the latice.

    Returns
    -------
    Tuple[Circuit, QuantumCircuit]
        The circuits, one for the mps/ttn simulator and one for qiskit.
    """
    sites = rows ** 2
    sycamore = Circuit(sites, D)
    qqc = QuantumCircuit(sites)

    for step in _create_pattern(rows):
        apply_single_gates(sycamore, qqc, sites)
        for pair in step:
            sycamore.append_gate(CircuitGate(ROOT_ISWAP, pair, 4))
            qqc.append(Q_ROOT_ISWAP, pair)
    apply_single_gates(sycamore, qqc, sites)
    return sycamore, qqc


def _create_pattern(rows: int) -> List[List[int]]:
    a_gates, b_gates, c_gates, d_gates = [], [], [], []
    for row in range(0, rows):
        if row < rows - 1:
            a_gates += [
                [x, x + rows]
                for x in range(
                    rows * row if row % 2 == 0 else rows * row + 1,
                    (row + 1) * rows,
                    2,
                )
            ]
            b_gates += [
                [x, x + rows]
                for x in range(
                    rows * row if row % 2 == 1 else rows * row + 1,
                    (row + 1) * rows,
                    2,
                )
            ]
        c_gates += [
            [x, x + 1]
            for x in range(
                rows * row if row % 2 == 0 else rows * row + 1,
                (row + 1) * rows - 1,
                2,
            )
        ]
        d_gates += [
            [x, x + 1]
            for x in range(
                rows * row if row % 2 == 1 else rows * row + 1,
                (row + 1) * rows - 1,
                2,
            )
        ]
    return [a_gates, b_gates, c_gates, d_gates, c_gates, d_gates, a_gates, b_gates]
