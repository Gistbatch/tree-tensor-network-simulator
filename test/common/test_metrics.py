import unittest

import numpy as np

from common.metrics import overlap_error


class TestMetrics(unittest.TestCase):
    def test_equality(self) -> None:
        ground_state = np.array([1, 0])
        self.assertEqual(overlap_error(ground_state, ground_state), 0)

    def test_opposite(self) -> None:
        ground_state = np.array([1, 0])
        approx = np.array([0, 1])
        self.assertEqual(overlap_error(approx, ground_state), 1)
        approx = np.array([0, 1j])
        self.assertEqual(overlap_error(approx, ground_state), 1)

    def test_mixed(self) -> None:
        ground_state = np.array([1, 0, 0, 0])
        approx = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        self.assertAlmostEqual(overlap_error(approx, ground_state), 1 - 1 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
