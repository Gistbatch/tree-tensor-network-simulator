import unittest

from ttn_circuit_sim.structure.qpair import QPair


class TestQPair(unittest.TestCase):
    def test_lt(self) -> None:
        a = QPair((1, 2), 2.0)
        b = QPair((1, 3), 2.0)
        c = QPair((2, 3), 1.0)
        self.assertLess(a, b)
        self.assertLess(a, c)
        self.assertLess(b, c)
