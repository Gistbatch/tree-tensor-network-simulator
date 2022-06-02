"""Implements common gates."""
from qiskit.extensions import UnitaryGate
from scipy.linalg import sqrtm
import numpy as np

# 1-qubit gates
H = np.array([[1.0, 1], [1, -1]]) / np.sqrt(2)
X = np.array([[0.0, 1], [1, 0]])
Y = np.array([[0.0, -1j], [1j, 0]])
W = (X + Y) / np.sqrt(2)

# 2-qubit gates
ROOT_X = sqrtm(X)
ROOT_Y = sqrtm(Y)
ROOT_W = sqrtm(W)

ISWAP = np.array(
    [[1.0, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)
ROOT_ISWAP = sqrtm(ISWAP)
CNOT = np.identity(4)[[0, 1, 3, 2]]
GATE_SET = [ROOT_W, ROOT_X, ROOT_Y]

# Qiskit definitions
Q_ROOT_ISWAP = UnitaryGate(ROOT_ISWAP, "siSWAP")
Q_ROOT_W = UnitaryGate(ROOT_W, "sW")
Q_ROOT_X = UnitaryGate(ROOT_X, "sX")
Q_ROOT_Y = UnitaryGate(ROOT_Y, "sY")

Q_GATE_SET = [Q_ROOT_W, Q_ROOT_X, Q_ROOT_Y]

SEED = 0
np.random.seed(SEED)
