"""Implements TTN contraction."""
import logging

import numpy as np
import opt_einsum as oe

from common import Tensor
from ..tnode import TNode


def contract(node: TNode, nrm: float, enable_gpu: bool = True) -> Tensor:
    """
    Fully contracts the node to a state vector.

    Contracts by recursively contracting child nodes depth first.

    Parameters
    ----------
    node: TNode
        The current node to contract.
    nrm: float
        Normalization parameter.
    enable_gpu: bool, default = True
        Use gpu to contract tensor.

    Returns
    -------
    Tensor
        Contractend tensor of the form (2,) * #Children + (parent_dim)
    """
    if node.is_leaf:
        # leaves are already contracted
        return node.tensor
    # add current parent tensor and label children [0,...,ndim]
    counter = node.tensor.ndim
    params = [node.tensor, list(range(counter))]
    for idx, child in enumerate(node.children):
        # add each contracted child tensor
        child_tensor = contract(child, 1, enable_gpu)
        # and place its already contracted leaves correctly
        child_indices = list(range(counter, counter + child_tensor.ndim - 1)) + [idx]
        counter += child_tensor.ndim - 1
        params += [child_tensor, child_indices]
    if not node.is_root:
        # keep parent index
        params[1][-1] = counter
    else:
        # or add normalization factor
        params += [np.array([nrm]), [node.tensor.ndim - 1]]

    logging.debug("Contracting node %s", node.name)
    if enable_gpu:
        import cupy as cp

        try:
            # GPU contraction using opt_einsum and cupy
            params = [
                cp.array(x) if idx % 2 == 0 else x for idx, x in enumerate(params)
            ]
            result = cp.asnumpy(oe.contract(*params, backend="cupy"))
        except:
            # oom or other errors on gpu
            result = np.einsum(*params)
        finally:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    else:
        result = np.einsum(*params)
    return result
