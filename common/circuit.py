"""Implements circuit and circuitgate classes."""
from dataclasses import dataclass
from typing import List, Optional

from .util import Tensor


@dataclass
class CircuitGate:
    """
    Circuit gate consisting of an unitary matrix and a list of sites the gate acts on.

    Attributes
    ----------
    gate_matrix: Tensor
        The gate matrix.
    sites: List[int]
        The affected sites.
    dim: Optional[int]
        The dimensionality in terms of singular values.
    """

    gate_matrix: Tensor
    sites: List[int]
    dim: Optional[int] = None

    def __post_init__(self):
        assert len(self.sites) >= 1
        if self.dim is None:
            self.dim = self.gate_matrix.shape[0]


@dataclass
class Circuit:
    """
    Quantum circuit represented as list of circuit gates.

    Attributes
    ----------
    l_sites: int
        The total number of sites.
    local_dimension: int
        The dimensionality of each side.
    """

    def __init__(self, l_sites: int, local_dimension: int) -> None:
        """
        Initialize a quantum circuit with `L` sites (qudits) and local dimension `d`.
        """
        self.l_sites = l_sites
        self.local_dimension = local_dimension
        self.gates: List[Tensor] = []

    def append_gate(self, gate: CircuitGate) -> None:
        """
        Append a quantum gate to the circuit.

        Parameters
        ----------
        gate: CircuitGate
            The gate to add.
        """
        for i in gate.sites:
            assert 0 <= i < self.l_sites
        self.gates.append(gate)
