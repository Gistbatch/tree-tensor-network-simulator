import unittest
import logging

import numpy as np

from ttn_circuit_sim import TNode, contract


class TestContract(unittest.TestCase):
    def test_contract(self) -> None:
        node = TNode(0, np.array([[1, 0]]))
        target = np.array([[1, 0]])
        self.assertTrue(np.all(contract(node, nrm=2) == target))
        node = TNode(
            "root",
            np.array([[[0], [1]]]),
            children=[
                TNode(
                    "0.0",
                    np.array([[1]]),
                    children=[
                        TNode(0, np.array([[1, 0]]).T),
                        TNode(1, np.array([[1, 0]]).T),
                    ],
                ),
                TNode(2, np.array([[1, 0], [0, 1]]).T),
            ],
        )
        target = np.array([0, 1, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(contract(node, nrm=1) == target.reshape(2, 2, 2)))
        try:
            self.assertTrue(
                np.all(
                    contract(node, nrm=1, enable_gpu=True) == target.reshape(2, 2, 2)
                )
            )
        except ModuleNotFoundError:
            logging.warning("Could not test GPU")
        self.assertTrue(np.all(contract(node, nrm=2) == 2 * target.reshape(2, 2, 2)))
