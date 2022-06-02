from dataclasses import asdict
import json
import os
import unittest

from circuits import create_structure_circuit, structure
from data.data import _sim_once, simulate, Result


class TestData(unittest.TestCase):
    def tearDown(self) -> None:
        if os.path.exists("data/test.json"):
            os.remove("data/test.json")

    def test_sim_once(self) -> None:
        results = _sim_once(
            lambda: create_structure_circuit(3),
            compress=False,
            dry_run=True,
            setting="qr",
            contract=False,
            simulate_mps=True,
        )
        self.assertEqual(len(results), 2)
        actual = asdict(results[0])
        actual.pop("apply")
        target = asdict(Result(0, 3, 13, False, 0, 0, 0, 420.0, 2 ** 3, 5, 192, "qr"))
        self.assertDictContainsSubset(actual, target)
        actual = asdict(results[1])
        actual.pop("apply")
        target = asdict(
            Result(-1, 0, 13, False, 0, 0, 0, 2452.0, 2 ** 5, 1, 192, "mps")
        )
        self.assertDictContainsSubset(actual, target)
        results = _sim_once(
            lambda: create_structure_circuit(3), contract=False, structures=[2]
        )
        actual = asdict(results[0])
        self.assertGreater(actual.pop("find"), 0)
        actual.pop("apply")
        target = asdict(
            Result(0, 2, 13, True, 0, 0, 0, 5576.0, 2 ** 6, 9, 192, "default")
        )
        self.assertDictContainsSubset(actual, target)

    def test_simulate(self) -> None:
        results = simulate([3])
        self.assertEqual(len(results), 1)
        target = asdict(
            Result(0, 3, 13, True, 0, 0, 0, 420.0, 2 ** 3, 5, 192, "default")
        )
        actual = asdict(results[0])
        actual.pop("apply")
        actual.pop("contract")
        self.assertDictContainsSubset(actual, target)
        results = simulate([3], "test", True)

        self.assertIsNone(results)
        with open("data/test.json", "r") as fp:
            results = json.load(fp)
        self.assertEqual(len(results), 1)
        actual = results[0]
        actual.pop("apply")
        actual.pop("contract")
        target = asdict(
            Result(0, 3, 9, True, 0, 0, 0, 756.0, 2 ** 3, 3, 105, "default")
        )
        self.assertDictContainsSubset(actual, target)
