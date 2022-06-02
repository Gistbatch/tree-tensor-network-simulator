import numpy as np
from qiskit import QuantumCircuit, Aer, assemble

import common
import mps_circuit_sim as mcs
import ttn_circuit_sim as tcs


def main():

    d = 2

    # start from computational basis state
    psi = mcs.MPS.basis_state(d, [0, 0, 0])
    L = psi.nsites
    print("number of qubits:", L)

    # define some standard gates
    H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    Ucnot = np.identity(4)[[0, 1, 3, 2]]

    # use circuit to prepare the Greenberger-Horne-Zeilinger (GHZ) state
    circ = common.Circuit(L, d)
    circ.append_gate(common.CircuitGate(H, [0]))
    circ.append_gate(common.CircuitGate(Ucnot, [0, 1], 2))
    circ.append_gate(common.CircuitGate(Ucnot, [0, 2], 2))

    mcs.apply_circuit(psi, circ, compress=True)

    print("output state as vector (should be the GHZ state):")
    print(psi.as_vector())

    psi = tcs.TTN.basis_state(d, [0, 0, 0], circ=circ)
    tcs.apply_circuit(psi, circ, compress=True)

    print("output state as vector (should be the GHZ state):")
    print(psi.as_vector())

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    svsim = Aer.get_backend("statevector_simulator")
    final_state = svsim.run(qc).result().get_statevector()
    print("output state as vector (should be the GHZ state):")
    print(final_state)


if __name__ == "__main__":
    main()
