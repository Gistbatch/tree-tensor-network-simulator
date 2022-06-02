import unittest

import numpy as np

from ttn_circuit_sim import TNode, count_dimensions


class TestCountDimensions(unittest.TestCase):
    def test_count_dimensions(self) -> None:
        node = TNode(0, np.array([[1, 0]]))
        self.assertTupleEqual(count_dimensions(node), (2.0, 2))
        node = TNode(
            "root",
            np.array([[[1], [1]]]),
            children=[
                TNode(
                    "0.0",
                    np.array([[1]]),
                    children=[
                        TNode(0, np.array([[1, 0]])),
                        TNode(1, np.array([[1, 0]])),
                    ],
                ),
                TNode(2, np.array([[1, 0], [0, 1]])),
            ],
        )

        self.assertTupleEqual(count_dimensions(node), (11.0, 2))
