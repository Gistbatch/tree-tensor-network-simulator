"""Implements creation of a ttn from single states."""
from typing import List


import numpy as np


from .snode import SNode
from ..tnode import PseudoTNode, TNode


def single_states_to_tree(
    single_states: List[int],
    local_dimension: int,
    structure: SNode,
) -> TNode:
    """Wrapper to create a Tree Tensor for a given circuit.

    Parameters
    ----------
    single_states: List[int]
        The individual basis states.
    local_dimension: int
        Dimensionality of individual leaves.
    structure: SNode
        The desired tree structure.

    Returns
    -------
    TNode
        The base state as Tree Tensor.
    """
    return _create_tree_level(structure, single_states, local_dimension)


def _create_tree_level(
    node: SNode, single_states: List[int], local_dimension: int
) -> TNode:
    """Recursively creates the tree bottom up.

    Leaves are created by assigning a tensor with dimension d
    and setting the corret initial state.
    Intermediate nodes are initialized with corret shape based on the provided structure,
    i. e. (child,child,...,child,parent), indices are tracked in a dict.
    The intermediate nodes get attached bottom up until the tree structure is reached.

    Parameters
    ----------
    root: SNode
        The structure defining tree.
    single_states: List[int]
        The individual basis states.
    local_dimension: int
        The dimensionality of the leaf nodes.

    Returns
    -------
    TNode
        The basis state as Tree Tensor.
    """
    if node.is_leaf:  # adds base state for leaf nodes
        tensor = np.zeros((local_dimension, 1))
        tensor[single_states[int(node.name)], 0] = 1
        new_node = TNode(node.name, tensor)
        return new_node

    tensor = np.array([1])
    # one index per child + one for the parent node
    shape = (1,) * (len(node.children) + 1)
    tensor = tensor.reshape(shape)
    child_nodes = []
    leaf_indices = {}  # dict to map child indices to leaves
    for idx, child in enumerate(node.children):
        # go through existing structure depth first
        new_child = _create_tree_level(child, single_states, local_dimension)
        if new_child.is_leaf:
            leaf_indices[new_child.name] = idx
        else:  # add all entries of the child to own dict and set the index
            leaf_indices.update([(k, idx) for k, _ in new_child.leaf_indices.items()])
        child_nodes.append(new_child)
    new_node = TNode(node.name, tensor, children=child_nodes, leaf_indices=leaf_indices)
    return new_node


def create_pseudo_tree(
    node: SNode, single_states: List[int], local_dimension: int
) -> PseudoTNode:
    """Recursively creates the tree bottom up.

    Creates a tree without real tensors similar to create_tree.

    Parameters
    ----------
    node: SNode
        The structure defining tree.
    single_states: List[int]
        The individual basis states.
    local_dimension: int
        The dimensionality of the leaf nodes.

    Returns
    -------
    PseudoTNode
        The basis state as Tree Tensor for dry suns.
    """
    if node.is_leaf:  # adds base state for leaf nodes
        shape = (local_dimension, 1)
        new_node = PseudoTNode(node.name, shape)
        return new_node

    # one index per child + one for the parent node
    shape = (1,) * (len(node.children) + 1)
    child_nodes = []
    leaf_indices = {}  # dict to map child indices to leaves
    for idx, child in enumerate(node.children):
        new_child = create_pseudo_tree(child, single_states, local_dimension)
        if new_child.is_leaf:
            leaf_indices[new_child.name] = idx
        else:
            leaf_indices.update([(k, idx) for k, _ in new_child.leaf_indices.items()])
        child_nodes.append(new_child)
    new_node = PseudoTNode(  # only the shape is necessary
        node.name, shape, children=child_nodes, leaf_indices=leaf_indices
    )
    return new_node
