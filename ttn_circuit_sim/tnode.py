"""Tree tensro"""
from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from anytree import NodeMixin
from anytree.cachedsearch import find_by_attr
from anytree.walker import Walker
import numpy as np
import opt_einsum as oe

from common import Tensor


class TNode(NodeMixin):
    """Tree tensor nodes.

    Represents the quantum state as a tree.

    Attributes
    ----------
    name: str
        Node name.
    tensor: Tensor
        Tensor of the node.
    parent: Tuple[TNode]
        Parent node.
    children: Tuple[TNode]
        Children nodes.
    leaf_indices: Dict[int, int]
        Mapping from tensor index to qubits.
    """

    walker = Walker()

    def __init__(
        self,
        name: str,
        tensor: Tensor,
        parent: Tuple = None,
        children: Tuple = None,
        leaf_indices: Dict[int, int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.tensor = tensor
        self.local_dim = tensor.shape[0]
        self.parent = parent
        self.leaf_indices = leaf_indices if leaf_indices is not None else {}
        self.tmp_dim = 0
        self.tmp_index = -1
        self.tmp_factor: Optional[Tensor] = None
        if children:
            self.children = children

    def __repr__(self) -> str:
        """
        Output formatting.
        """
        return f"Node: {self.name}, Tensor shape: {self.tensor.shape}"

    def __getitem__(self, key) -> Union[TNode, List[TNode]]:
        """Indexing wrapper to simplify node access.

        Should only be invoked on the tree root.

        Examples
        --------
        root[i]: finds and returns leaf i.
        root[i:]: finds leaf i and returns all nodes from i to the root.
        root[i:j]: finds leaf i and leaf j. Returns all nodes on the path from i to j.
        """
        if isinstance(key, int):
            return find_by_attr(self, key)
        if isinstance(key, slice):
            start = find_by_attr(self, key.start)
            if key.stop:
                stop = find_by_attr(self, key.stop)
            else:
                stop = self
            upw, common, down = self.walker.walk(start, stop)
            return list(upw) + [common] + list(down)
        return []

    def apply_gate(self, gate_matrix: Tensor) -> None:
        """Updates the leaf tensor by applying a tensor.

        Single particle gates can be immediately contracted without increasing dimension.
        T = T @ G. The shape stays the same.

        Parameters
        ----------
        gate_matrix: Tensor
            The (2,2) gate matrix.
        """
        assert self.is_leaf
        self.tensor = np.einsum("ab, bc", gate_matrix, self.tensor)

    def apply_gate_and_reshape(self, update: Tensor) -> None:
        """Updates the leaf tensor by applying a tensor and reshaping afterwards.

        Two particle gates are applied with factors resulting from svd.
        The gate comes in as (physical, parent, gate).
        Multiply along parent dimension to attach it.
        The gate dimension is routed to the parent.
        The resulting shape is (physical, parent*gate).

        Parameters
        ----------
        update: Tensor
            The (2,2,n) matrix from the split gate matrix.
        """
        assert self.is_leaf
        self.tensor = np.einsum("abc, bd", update, self.tensor)
        self.tensor = self.tensor.reshape(self.local_dim, -1)

    def update(self, gate_dim: int, site_i: int, site_j: int) -> None:
        """Updates the intermediate tensors by wiring additional dimensions.

        Tensor dimensions on affected indices are increased by the gate dimension.
        This is done by applying an outer product to the correct dimensions:
        G = G x I similar to kron.

        Two cases for the dimensions:
            - Common ancestor of both leaves (...,i,...,j,...)
            - Only one leaf affected. (...,i,...,j)
        In the new shape two dimensions will be increased by gate_dim.

        Parameters
        ----------
        gate_dim: int
            The number of singular values in the gate.
        site_i: int
            The first affected qubit.
        site_j: int
            The second affected qubit.
        """
        assert not self.is_leaf
        if self.is_root:
            self.tmp_index = site_i
            self.tmp_dim = gate_dim
            return
        # find the indices to increase
        # either two children or parent + child
        ij_indices = np.array(
            [
                self.leaf_indices.get(site_i, self.tensor.ndim - 1),
                self.leaf_indices.get(site_j, self.tensor.ndim - 1),
            ]
        )
        shape = list(self.tensor.shape)
        indices = list(range(1, self.tensor.ndim * 2 + 1, 2))
        for idx in ij_indices:
            shape[idx] *= gate_dim
        self.tensor = oe.contract(  # einsum is actually faster than kron
            np.identity(gate_dim), 2 * ij_indices, self.tensor, indices
        ).reshape(shape)


class PseudoTNode(NodeMixin):
    """Drop in replacement for TNode in dry runs.

    Just keeps track of the shapes instead of actual tensors.

    Attributes
    ----------
    name: str
        Node name.
    shape: Tuple[int]
        Supposed shape.
    parent: Tuple[TNode]
        Parent node.
    children: Tuple[TNode]
        Children nodes.
    leaf_indices: Dict[int, int]
        Mapping from tensor index to qubits.
    """

    pseudo_walker = Walker()

    def __init__(
        self,
        name: str,
        shape: Tuple[int],
        parent: Tuple = None,
        children: Tuple = None,
        leaf_indices: Dict[int, int] = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.shape = shape
        self.local_dim = shape[0]
        self.parent = parent
        self.leaf_indices = leaf_indices if leaf_indices is not None else {}
        self.tmp_dim = 0

        if children:
            self.children = children

    @property
    def tensor(self):
        """Local attribute to ensure correctness."""
        return PseudoTensor(self.shape)

    def __repr__(self) -> str:
        """
        Output formatting.
        """
        return f"Pseudo Node: {self.name}, Tensor shape: {self.shape}"

    def __getitem__(self, key) -> Union[PseudoTNode, List[PseudoTNode]]:
        """
        Indexing wrapper to simplify node access.

        Should only be invoked on the tree root.
        Examples
        --------
        root[i]: finds and returns leaf i.
        root[i:]: finds leaf i and returns all nodes from i to the root.
        root[i:j]: finds leaf i and leaf j. Returns all nodes on the path from i to j.
        """
        if isinstance(key, int):
            return find_by_attr(self, key)
        if isinstance(key, slice):
            start = find_by_attr(self, key.start)
            if key.stop:
                stop = find_by_attr(self, key.stop)
            else:
                stop = self
            upw, common, down = self.pseudo_walker.walk(start, stop)
            return list(upw) + [common] + list(down)
        return []

    def apply_gate(self, gate_matrix: Tensor) -> None:
        """Updates the leaf tensor by applying a tensor.

        Single particles don't change the shape -> d nothing.

        Parameters
        ----------
        gate_matrix: Tensor
            The (2,2) gate matrix.
        """
        assert self.is_leaf

    def apply_gate_and_reshape(self, update: Tensor) -> None:
        """Updates the leaf shape by simulating a gate.

        Parameters
        ----------
        update: Tensor
            The update tensor with correct shape.
        """
        assert self.is_leaf
        gate_dim = update.shape[-1]
        self.shape = self.shape[:-1] + (self.shape[-1] * gate_dim,)

    def update(self, gate_dim: int, site_i: int, site_j: int) -> None:
        """Updates the intermediate tensors.

        Tensor dimensions on affected indices are increased by the gate dimension.

        Parameters
        ----------
        gate_dim: int
            The number of singular values in the gate.
        site_i: int
            The first affected qubit.
        site_j: int
            The second affected qubit.
        """
        assert not self.is_leaf
        index_i = self.leaf_indices.get(site_i, len(self.shape) - 1)
        index_j = self.leaf_indices.get(site_j, len(self.shape) - 1)

        shape = (
            self.shape[: min(index_i, index_j)]
            + (self.shape[min(index_i, index_j)] * gate_dim,)
            + self.shape[min(index_i, index_j) + 1 : max(index_i, index_j)]
            + (self.shape[max(index_i, index_j)] * gate_dim,)
            + self.shape[max(index_i, index_j) + 1 :]
        )
        self.shape = shape


class PseudoTensor(NamedTuple):
    """Helper to  make tensor.shape calls possible."""

    shape: Tuple[int]
