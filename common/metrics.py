"""Implements common metrics to compare quantum states."""
import numpy as np

from .util import Tensor


def overlap_error(approx: Tensor, ground_state: Tensor) -> float:
    r"""Calculates the overlap error between two quantum states.

    \mathcal{E}_{\text{overlap}} = 1 - \braket{\psi_{\text{tree}} \vert \psi_{\text{exact}}}

    Parameters
    ----------
    approx: Tensor
        First state.
    ground_state: Tensor
        Second state.

    Returns
    -------
    float
        The calculated error.
    """
    assert approx.ndim == 1
    assert ground_state.ndim == 1
    return np.real(1 - approx @ ground_state.conj().T)
