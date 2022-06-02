"""Data creation."""
from dataclasses import dataclass, asdict
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple
import json
import logging
import time
import traceback


from qiskit import Aer, QuantumCircuit

from circuits import create_structure_circuit, create_sycamore_circuit
from common import Circuit, overlap_error
import mps_circuit_sim as mcs
import ttn_circuit_sim as tcs


@dataclass
class Result:
    """Result tracking."""

    x: int
    clusters: int
    qubits: int
    compress: bool
    find: float
    apply: float
    contract: float
    sum_prod_dims: float
    max_dim: int
    max_leaves: int
    gates: int
    setting: str = "intel"


def _sim_once(
    circuit_func: Callable[[int], Tuple[Circuit, QuantumCircuit]], **kwargs
) -> List[Result]:
    circ, qcirc = circuit_func()
    logging.basicConfig(level=logging.INFO)
    logging.info("number of qubits: %d", circ.l_sites)
    contract = kwargs.get("contract", True)
    dry_run = kwargs.get("dry_run", False)
    if contract:
        svsim = Aer.get_backend("statevector_simulator")
        final_state = svsim.run(qcirc).result().get_statevector()
        logging.info("output state as vector reference")
        logging.info(final_state)

    results, timings = [], []
    for idx, structure in enumerate(kwargs.get("structures", [None])):
        logging.info("Run with param=%d", idx)
        timings.append(time.perf_counter())
        if isinstance(structure, int):
            structure = tcs.find_tree_structure(
                circ, clusters=structure, find_state=True
            )
        psi = tcs.TTN.basis_state(
            circ.local_dimension,
            [0] * circ.l_sites,
            structure=structure,
            circ=circ,
            **kwargs,
        )
        timings.append(time.perf_counter())
        try:
            tcs.apply_circuit(psi, circ, compress=kwargs.get("compress", True))
            timings.append(time.perf_counter())
            if contract:
                vector = psi.as_vector()
        except:
            logging.warning("Error on run=%d", idx)
            traceback.print_exc()
        else:
            timings.append(time.perf_counter())
            cumulative, max_bond = psi.bond_data()
            if contract:
                vector = vector.reshape((2,) * circ.l_sites).transpose().reshape(-1)
                error = overlap_error(vector, final_state)
                logging.debug("Overlap error: %f", error)
                assert error < 1e-8
            results.append(
                Result(
                    idx,
                    len(psi.root.children),
                    circ.l_sites,
                    kwargs.get("compress", True),
                    timings[1] - timings[0]
                    if kwargs.get("structures", None) is not None
                    else 0,
                    timings[2] - timings[1],
                    timings[3] - timings[2] if contract else 0,
                    cumulative,
                    max_bond,
                    psi.max_leaves(),
                    len(circ.gates),
                    kwargs.get("setting", "default"),
                )
            )
    if kwargs.get("simulate_mps", False):
        timings = []
        timings.append(time.perf_counter())
        psi = mcs.MPS.basis_state(
            circ.local_dimension, [0] * circ.l_sites, dry_run=dry_run
        )
        mcs.apply_circuit(
            psi,
            circ,
            compress=kwargs.get("compress", True),
            dry_run=dry_run,
        )
        timings.append(time.perf_counter())
        if contract:
            vector = psi.as_vector()
            timings.append(time.perf_counter())
            vector = vector.reshape((2,) * circ.l_sites).transpose().reshape(-1)
            error = overlap_error(vector, final_state)
            assert error < 1e-8
        cumulative, max_bond = psi.bond_data()
        results.append(
            Result(
                -1,
                0,
                circ.l_sites,
                kwargs.get("compress", True),
                0,
                timings[1] - timings[0],
                timings[2] - timings[1] if contract else 0,
                cumulative,
                max_bond,
                1,
                len(circ.gates),
                "mps",
            )
        )

    return results


def simulate(
    units: Iterable[int],
    file: Optional[str] = None,
    use_sycamore: bool = False,
    **kwargs,
) -> Optional[List[Result]]:
    """
    Wrapper function to simulate sycamore or structure circuit.

    Does the following
    - Creates the respective circuit
    - Finds a structure or simulates all given structures
    - Applies the circuit to the ttn
    - Contracts to the full statevector
    - Simulates on the mps
    - Runs comparison on qiskit

    Parameters
    ----------
    units: Iterable[int]
        The number of qubits this simulation should be executed with.
    file: Optional[str], default = None
        File name to write to.
    use_sycamore: bool, default = False
        Switch desired circuits.

    kwargs:
        - contract: bool, default = True
            Whether to construct the full statevector.
        - dry_run: bool, default = False
            Only simulates gate application.
        - simulate_mps: bool, default = False
            Whether to compare to MPS.
        - setting: str, default = "default"
            Setting parameter in result dict.
        - compress: bool, default = True
            Switch between SVD and QR.
        - structures: Optional[Iterable[Union[SNode, int]]], default = None
            Initialization structures for the ttn.
            Either number of clusters or fixed structure nodes.

    Additional kwargs will be passed to TTN.basis_state().

    Returns
    -------
    Optional[List[Result]]
        The results as list if no file location is given else None.
    """
    results = []
    if use_sycamore:
        circ_func = create_sycamore_circuit
    else:
        circ_func = create_structure_circuit
    for unit in units:
        logging.info("Unit: %d", unit)
        results += _sim_once(partial(circ_func, unit), **kwargs)
    if file is not None:
        with open(f"data/{file}.json", "w+", encoding="utf-8") as out_file:
            json.dump([asdict(r) for r in results], out_file)
        return None
    return results


def _experiment1() -> None:
    # cluster sizes
    logging.info("Start experiment 1")
    logging.info("=" * 80)
    results = []
    poss = [True, False]
    for idx, experiment in enumerate(poss):
        for compress in poss:
            results += simulate(
                [4],
                use_sycamore=experiment,
                compress=compress,
                simulate_mps=True,
                structures=range(3, 11),
                setting="svd" if compress else "qr",
            )

        file = "structure" if idx == 1 else "sycamore"
        with open(f"data/{file}_experiment1.json", "w+", encoding="utf-8") as out_file:
            json.dump([asdict(r) for r in results], out_file)
        results = []


def _experiment2() -> None:
    # dry runs
    logging.info("Start experiment 2")
    logging.info("=" * 80)
    results = []
    configs = [
        {
            "setting": "default",
            "simulate_mps": True,
        },
        {"maximize_for_prod": True, "setting": "t1"},
        {"bound": 10, "setting": "bounded"},
        {"flat": True, "setting": "minimized_sum"},
    ]
    poss = [True, False]
    for idx, experiment in enumerate(poss):
        for conf in configs:
            logging.info("Simulate %s for experiment: sycamore=%s", conf, experiment)
            print("Simulate %s for experiment: sycamore=%s", conf, experiment)
            results += simulate(
                range(3, 11) if experiment else range(3, 23),
                use_sycamore=experiment,
                dry_run=True,
                contract=False,
                **conf,
            )

        file = "structure" if idx == 1 else "sycamore"
        with open(f"data/{file}_experiment2.json", "w+", encoding="utf-8") as out_file:
            json.dump([asdict(r) for r in results], out_file)
        results = []


def _experiment3() -> None:
    # structure without contraction
    logging.info("Start experiment 3")
    logging.info("=" * 80)
    simulate(
        range(5, 10),
        "structure_experiment3",
        use_sycamore=False,
        contract=False,
        simulate_mps=True,
        setting="svd",
    )


def create_data() -> None:
    """
    Top level function to execute all experiments.
    """
    logging.basicConfig(level=logging.DEBUG, filename="strc.log", filemode="w+")
    _experiment1()
    _experiment2()
    _experiment3()
