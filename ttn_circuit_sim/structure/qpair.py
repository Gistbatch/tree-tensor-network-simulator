"""Helper for similarity matrix."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QPair:
    """Unique pair of qubits with their similarity.

    Attributes
    ----------
    qubits: Tuple[int]
        Identifier of qubits in the pair.
    similarity: float
        Similarity of the two qubits.
    """

    qubits: Tuple[int]
    similarity: float

    def __lt__(self, other: QPair) -> bool:
        if self.similarity == other.similarity:
            return self.qubits < other.qubits
        return self.similarity >= other.similarity
