"""Implements sampling from a ttn."""
# TODO FIX ordering
from bisect import bisect_left
from itertools import accumulate
from typing import List

import numpy as np

from common import Tensor
from .contract import contract
from ..tnode import TNode

BASE_OP = [
    np.array([[1, 0], [0, 0]]),  # |0>
    np.array([[0, 0], [0, 1]]),  # |1>
]


def sample(root: TNode, nrm: float) -> Tensor:
    """
    Randomly samples a state from a tree tensor.

    Implementation from "Perfect sampling with unitary tensor networks".
    Samples by sequentially drawing qubits using the chain rule in combination with Born's rule.
    """
    current_state = np.eye(2) * np.abs(nrm) ** 2
    state_sample = _sample_and_contract(root, current_state)
    return np.array(state_sample)


def _sample_qubit(tensor: Tensor, shots: int = 1024) -> int:
    # calculate probabilities by contracting over observable and normalizing
    state_probabilities = []
    for operator in BASE_OP:
        state_probabilities.append(np.einsum("ab, ab", tensor, operator).real)
    state_probabilities *= 1.0 / sum(state_probabilities)

    # draw randomly
    result = [0, 0]
    for _ in range(shots):
        idx = _sample_once(state_probabilities)
        result[idx] += 1
    result = np.argmax(result)
    return result


def _sample_once(state_probabilities: List[float]) -> int:
    prefix_sum = accumulate(probability for probability in state_probabilities)
    val = np.random.uniform()
    return bisect_left(prefix_sum, val)


def _sample_and_contract(node: TNode, current_state: Tensor) -> List[int]:
    if node.is_leaf:
        tensor = node.tensor
        tensor = np.einsum("ab, bc, cd", tensor, current_state, tensor.conj().T)
        state = _sample_qubit(tensor)
        node.tensor = node.tensor.take(state, axis=1)  # TODO check correctness
        return [state]
    states = []
    for child in node.children:
        states += _sample_and_contract(child, current_state)
        child_tensor = child.tensor
        current_state = np.einsum(
            "abcd, c, d",
            np.expand_dims(current_state, axis=(2, 3)),
            child_tensor,
            child_tensor.conj().T,
        )
    node.tensor = contract(node, nrm=1)  # TODO check if last contracted is correct
    return states
