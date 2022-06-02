"""Implements the structure search algorithm."""
from collections import Counter
from typing import List, Optional
import logging
import math

from sklearn.cluster import SpectralClustering
import numpy as np


from common import Circuit, Tensor
from .qpair import QPair
from .snode import SNode


def find_tree_structure(
    circuit: Circuit,
    **kwargs,
) -> SNode:
    r"""
    Finds a tree structure based on qubit similarity.

    Qubit similarity is calculated by shared gates:
    d_{qc}(q_1,q_2) = |G(q_1) \cap G(q_2)| + \frac{1}{|G(q_1)|+|G(q_2)|}
    Structure is first determined by clustering (partitioning) the circuit
    in similar sized subtrees with high count of intra-cluster gates.
    Each subtree is then created bottom up, based on similarity or as single
    flat node.

    Parameters
    ----------
    circuit: Circuit
        The circuit definition.
    kwargs:
        - clusters: int
            Desired number of clusters.
        - random_state: int
            For the clustering initialization.
        - d_max: int
            Desired d_max parameter (should fit clusters).
        - find_state: bool
            Find a random_state which yields similar sized clusters.
        - flat: bool, default = False
            Change to single node subtrees.

    Returns
    -------
    SNode
        The final tree structure.
    """
    if circuit.l_sites == 1:
        return SNode(0)
    # setup
    clusters = kwargs.get("clusters", math.ceil(circuit.l_sites / 8))
    random_state = kwargs.get("random_state", None)
    d_max = kwargs.get("d_max", math.ceil(circuit.l_sites / clusters))
    find_state = kwargs.get("find_state", None)
    # precomputed similarity matrix necessary for clustering
    similarity = _to_similarity_matrix(circuit)
    if random_state is not None:
        # Spectral clustering uses multiple k-means runs with random inits for good results
        clustering = SpectralClustering(
            n_clusters=clusters, affinity="precomputed", random_state=random_state
        ).fit(similarity)
    elif find_state:
        # find random state for similar sized clusters
        for idx in range(1000):
            clustering = SpectralClustering(
                n_clusters=clusters, affinity="precomputed", random_state=idx
            ).fit(similarity)
            if _max_cluster_size(clustering.labels_) <= d_max:
                logging.info("Used random state %d for clustering", idx)
                break
    else:
        clustering = SpectralClustering(
            n_clusters=clusters, affinity="precomputed"
        ).fit(similarity)
    children = []
    for idx in range(clusters):
        # create subtrees based on cluster labels
        if kwargs.get("flat", False):
            subtree = _create_subtree_flat(np.where(clustering.labels_ == idx)[0], idx)
        else:
            subtree = _create_subtree(
                np.where(clustering.labels_ == idx)[0], similarity, idx
            )
        if subtree is not None:
            children.append(subtree)

    return SNode("root", children=children)


def _to_similarity_matrix(circuit: Circuit) -> Tensor:
    """Creates similarity matrix based on the circuit.

    All gates contribute to two counts:
        - Individual for each qubit
        - Shared gates

    Based on those the similarity is calculated.

    Parameters
    ----------
    circuit: Circuit
        The circuit definition.

    Returns
    -------
    Tensor
        (n_qubits,n_qubits) shaped similarity matrix.
    """
    similarity = np.eye(circuit.l_sites) * (2 ** 63)  # self is similar
    pairwise_gate_count = Counter()
    total_gate_count = Counter()
    for gate in circuit.gates:
        # count gates for similarity measure
        if len(gate.sites) == 2:
            pairwise_gate_count[(gate.sites[0], gate.sites[1])] += gate.dim
            total_gate_count[gate.sites[0]] += gate.dim
            total_gate_count[gate.sites[1]] += gate.dim
    for (first, second), gate_count in pairwise_gate_count.items():
        # similarity definition
        sim = gate_count + 1 / (total_gate_count[first] + total_gate_count[second])
        # fill in entries
        similarity[first][second] = sim
        similarity[second][first] = sim

    return similarity


def _create_subtree(
    leaves: List[int], similarity: Tensor, cluster: int
) -> Optional[SNode]:
    """Creates the tree for each cluster based on similarity, bottom up.

    More similar nodes are grouped on lower levels of the trees.
    Qubits with same similarity values are put on same tree level.
    For a new value a new tree level is created.

    Parameters
    ----------
    leaves: List[int]
        List of qubits in the cluster.
    similarity: Tensor
        The similarity matrix.
    cluster: int
        Cluster label.

    Returns
    -------
    SNode
        The structure for the subtree.
    """
    if len(leaves) == 0:
        return None
    if len(leaves) == 1:
        return SNode(leaves[0])
    entries = []
    # find relevant qubits and similarity
    for idx, i in enumerate(leaves):
        for j in leaves[idx + 1 :]:
            sim = similarity[i][j]
            entries.append(QPair((i, j), sim))

    # sort by similarity
    entries = sorted(entries)
    seen = set()  # keep up with already created nodes
    counter = len({e.similarity for e in entries if e.similarity > 0}) - 1  # labels
    current = SNode(f"{cluster}.{counter}")  # current level
    sim = entries[0].similarity  # started with highest similarity
    for idx, entry in enumerate(entries):
        i, j = entry.qubits
        leaves = []
        # add unseen qubits to tree
        if i not in seen:
            leaves.append(SNode(i))
        if j not in seen:
            leaves.append(SNode(j))
        if not leaves:
            continue
        # same similarity on same level
        if sim == entry.similarity and len(leaves) == 1 or len(current.children) == 0:
            current.children = list(current.children) + leaves
        # create a new tree level if not
        else:
            counter -= 1
            current = SNode(f"{cluster}.{counter}", children=[current] + leaves)
        seen.update({i, j})
        sim = entry.similarity
    return current


def _max_cluster_size(labels: List[int]) -> int:
    """
    Finds the size of the biggest cluster.

    Parameters
    ----------
    labels: List[int]
        The cluster labels.

    Returns
    -------
    int
        The maximal number.
    """
    count = Counter(labels)
    return count.most_common(1)[0][1]


def _create_subtree_flat(leaves: List[int], cluster: int) -> SNode:
    """
    Creates the tree for each cluster as one big node.

    Parameters
    ----------
    leaves: List[int]
        List of qubits in the cluster.
    cluster: int
        Cluster label.

    Returns
    -------
    SNode
        The structure for the subtree.
    """
    if len(leaves) == 0:
        return None
    if len(leaves) == 1:
        return SNode(leaves[0])
    children = []
    for leave in leaves:
        children.append(SNode(leave))
    return SNode(f"{cluster}.0", children=children)
