"""Data tracking."""
from typing import Tuple

from common.util import dim_product
from ..tnode import TNode


def count_dimensions(node: TNode) -> Tuple[float, int]:
    r"""Counts the dimensions.

    Finds the total number of necessary entries and the maximum bond:

    B_{\max}=\max_{T \in TTN}\max_{i}|T_i|
    B_{\text{sum}}=\sum_{T \in \text{TTN}} \prod_{i}|T_i| = E

    Parameters
    ----------
    node: TNode
        The tree to search.

    Returns
    -------
    Tuple[float, int]
        (maximum bond dimension, sum over product of bond dimensions)
    """
    if node.is_leaf:
        return dim_product(node.tensor.shape), max(node.tensor.shape)
    count = dim_product(node.tensor.shape)
    current_max = int(max(node.tensor.shape))
    for child in node.children:
        tmp_count, tmp_max = count_dimensions(child)
        count += tmp_count
        current_max = int(max(current_max, tmp_max))
    return count, current_max
