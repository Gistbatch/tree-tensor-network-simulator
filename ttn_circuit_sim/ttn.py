"""Wrapper class for TTN Simulation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from common import Circuit, Tensor
from .gate_ops import apply_circuit
from .ops import (
    count_dimensions,
    contract,
    contract_factor_on_index,
    orthonormalize_qr,
    precontract_root,
    orthonormalize_svd,
    pseudo_orthonormalize,
)
from .structure import (
    create_pseudo_tree,
    find_tree_structure,
    single_states_to_tree,
    SNode,
)
from .tnode import TNode


class TTN:
    """
    Base object holding the root node of the tree.

    Arguments
    ---------
    d: int
        Qubit dimension.
    root: TNode
        The root of the tree tensor.
    d_max: int, default = 2 ** 63
        Maximum allowed bond dimension in the tree.
    enable_gpu: bool, default = True
        Use gpu functions.
    """

    def __init__(
        self,
        local_dimension: int,
        root: TNode,
        d_max: int = 2 ** 63,
        enable_gpu: bool = True,
        dry_run: bool = False,
    ) -> None:
        self.local_dimension = local_dimension
        self.root = root
        self.nrm = 1.0
        self.d_max = d_max
        self.enable_gpu = enable_gpu
        self.dry_run = dry_run

    @property
    def local_dim(self):
        """
        Local (physical) dimension at each lattice site.
        """
        return self.local_dimension

    @property
    def nsites(self):
        """
        Number of lattice sites.
        """
        return len(self.root.leaves)

    @property
    def dtype(self):
        """
        Data type of tensor entries.
        """
        return self.root.tensor.dtype

    @classmethod
    def basis_state(
        cls,
        local_dimension: int,
        single_states: List[int],
        structure: Optional[SNode] = None,
        circ: Optional[Circuit] = None,
        **kwargs,
    ) -> TTN:
        """
        Represent a computational basis state as TTN;
        `single_states` contains the individual basis states for each site as list.

        In case no structure is defined, a structure search is executed,
        where multiple dry runs are executed to minimize a given metric.
        The tree initialization can be controlled as well.

        Arguments
        ---------
        local_dimension: int
            Qubit dimension.
        single_states: List[int]
            Basis states.
        structure: SNode
            The desired structure of the tree tensor.
        d_max: int, default = 2 ** 63
            Maximum allowed bond dimension in the tree.
        enable_gpu: bool, default = True
            Use gpu functions.
        dry_run: bool, default = True
            Only calculate on tensor dimensions.
        flat: bool, default = False
            Use flat subtree structure.
        bound: int, default = 0
            Set maximal cluster size.
        maximize_for_prod: bool, default = False
            Minimize for mix between max dim and product.

        Returns
        -------
        TTN
            The wrapper for the tree tensor network.
        """
        assert (
            structure is not None or circ is not None
        ), "Structure of Circuit have to be set"
        d_max = kwargs.get("d_max", 2 ** 63)
        if structure is None:  # need the circuit for structure search
            structures = {}
            length = len(single_states)
            if length > 1:  # filter edge case and find acceptable number of subtrees
                length = length // 2 if length > 4 else length + 1
                for clusters in range(2, min(length, 15)):
                    tmp_structure = find_tree_structure(
                        circ, clusters=clusters, find_state=True, **kwargs
                    )  # create a temporary structure given the current params
                    root = create_pseudo_tree(
                        tmp_structure, single_states, local_dimension
                    )  # create a ttn
                    psi = cls(local_dimension, root, d_max, False, dry_run=True)
                    apply_circuit(psi, circ)  # and perform a dry run
                    tmp_cumulative, max_bond = psi.bond_data()
                    logging.info(
                        "Cluster: %d Sum: %d Max: %d T1: %d, T2:%d",
                        clusters,
                        tmp_cumulative,
                        np.log2(max_bond),
                        tmp_cumulative * max_bond,
                        tmp_cumulative * np.log2(max_bond),
                    )
                    # keep the results necessary for the comparison
                    structures[tmp_structure] = _Structure(
                        clusters, max_bond, psi.max_leaves(), tmp_cumulative
                    )
                # select the best structure
                structure = _find_best_structure(structures, **kwargs)
            else:
                structure = find_tree_structure(circ)

        if kwargs.get("dry_run", False):
            root = create_pseudo_tree(structure, single_states, local_dimension)
            return cls(local_dimension, root, enable_gpu=False, dry_run=True)
        root = single_states_to_tree(single_states, local_dimension, structure)
        return cls(local_dimension, root, d_max, kwargs.get("enable_gpu", False))

    def as_vector(self) -> Tensor:
        """Contracts the tensort network into a single node.

        The vectors from contraction has to be ordered again.

        Returns
        -------
        Tensor
            The fully contracted ttn.
        """
        if self.dry_run:  # just give any vector
            return np.array([0] * (self.nsites - 1) + [1])
        # contract and add normalization factor
        state = contract(self.root, self.nrm, self.enable_gpu)
        # find current order
        axes = [int(node.name) for node in self.root.leaves]
        shape = (self.local_dim,) * len(axes)
        state = state.reshape(shape)  # form axis
        state = state.transpose(np.argsort(tuple(axes))).reshape(-1)  # sort
        return state

    def bond_data(self) -> Tuple[float, int]:
        """
        Finds the internal bond data.

        Returns
        -------
        Tuple[int]
            (maximum bond dimension, sum over product of bond dimensions)
        """
        return count_dimensions(self.root)

    def max_leaves(self) -> int:
        """
        Finds the maximum number of leaves assigned to a subtree of the root.

        Returns
        -------
        int
            The maximum.
        """
        return max(len(child.leaves) for child in self.root.children)

    def orthonormalize(
        self, site_i: int, site_j: int, compress: bool = False, tol: float = 0.0
    ) -> None:
        """Orthonormalize the TTN using SVDs or QR decompositions.

        Root is chosen as as central tensor, i. e. all normalization factors are pushed to the root.
        A normalization value is than stored in the ttn.
        Start with one leaf from the bottom until the first common ancestor of both leaves is reached.
        Stop there as we only need to normalize once and start with the second leaf bottom up.
        All normalizaion factors a checked if it is possible to early stop.

        Parameters
        ----------
        site_i: int
            The first affected qubit.
        site_j: int
            The second affected qubit.
        compress: bool, default = False
            Use compression in normalization.
        tol: float, default = 0.0
            The compression tolerance.
        """
        # select method
        if self.dry_run:  # update shapes only
            local_orthonormalize = pseudo_orthonormalize
        elif compress:  # needs tolerance and truncation params
            local_orthonormalize = lambda A, x, T: orthonormalize_svd(
                A, x, tol, self.d_max, T
            )
        else:
            local_orthonormalize = orthonormalize_qr

        factor = None
        # start from i bottom up
        for node in self.root[site_i:]:
            if site_j in node.leaf_indices.keys():  #
                # early stop on the first common ancestor with j
                if self.dry_run:  # update already on dry run
                    node.shape = (
                        node.shape[: node.leaf_indices[site_i]]
                        + (factor,)
                        + node.shape[node.leaf_indices[site_i] + 1 :]
                    )
                elif node.is_root:
                    # don't contract on the root node for performance reasons
                    # see precontract_root
                    node.tmp_factor = factor
                else:  # stop early and simply contract without normalization
                    node.tensor = contract_factor_on_index(
                        node.tensor, factor, node.leaf_indices[site_i]
                    )
                break
            # actual normalization
            factor = local_orthonormalize(node, site_i, factor)
            if _is_square_identity(factor):
                break

        factor = None
        # normalize the rest this time from j up
        for node in self.root[site_j:]:
            if node.is_root and not node.tmp_dim == 0:
                # no early stopping due to common ancestor
                # root has to be treated differently
                precontract_root(node, site_j, factor)
                factor = None

            factor = local_orthonormalize(node, site_j, factor)
            if _is_square_identity(factor):
                break

        if not self.dry_run and factor is not None:
            self.nrm *= factor[0][0].real


def _is_square_identity(factor: Optional[Tensor]) -> bool:
    """Check if a tensor is a square identity.

    Parameters
    ----------
    factor: Optional[Tensor]
        The factor to be checked.
    Returns
    -------
    bool
        True if the tensor is a np.identity().
    """
    if (
        factor is None
        or isinstance(factor, int)
        or factor.ndim != 2
        or factor.shape[0] != factor.shape[1]
    ):
        return False
    return np.allclose(factor, np.identity(factor.shape[0]))


@dataclass
class _Structure:
    """Structure tracking."""

    clusters: int
    max_dim: int
    max_leaves: int
    sum_prod_dims: int


def _find_best_structure(structures: Dict[SNode, _Structure], **kwargs) -> SNode:
    """Returns the best structure given the parameters.

    Filters by the given parameters and chooses some sensible defaults.
    In the best case we actually fullfill the desired property:
    B_c < 2^{l_c} <= D_{\max}

    Parameters
    ----------
    structures: Dict[SNode, _Structure]
        The mapping of all tested initializations.
    kwargs:
        - bound: int, default = 0
            Restrict cluster size if possible.
        - maximize_for_prod: bool, default = false
            Try to maximize for the product of two factors

    Returns
    -------
    SNode
        The best structure.
    """
    bound = kwargs.get("bound", 0)
    maximize_for_prod = kwargs.get("maximize_for_prod", False)
    structures = sorted(  #  initially sort by default properties
        structures.items(),
        key=lambda s: (
            s[1].sum_prod_dims * (s[1].max_dim if maximize_for_prod else 1),
            s[1].clusters * s[1].max_leaves,
            -s[1].clusters,
        ),
    )
    if bound:  # filter out structures with big subtrees
        tmp = list(
            filter(
                lambda s: (s[1].max_dim < 2 ** bound),
                structures,
            )
        )
        # if none remain keep original list
        structures = tmp if tmp else structures
    return next(  # check for the desired property
        filter(
            lambda s: (s[1].max_dim < 2 ** s[1].max_leaves),
            structures,
        ),
        structures[0],  # if none fullfills give the previous best
    )[0]
