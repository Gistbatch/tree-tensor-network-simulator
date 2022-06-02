from tkinter import Y
import unittest

import matplotlib.pyplot as plt
import pandas as pd

from data.vis import _plot_property


class TestVis(unittest.TestCase):
    def test_plot_property(self) -> None:
        data = pd.DataFrame(
            [
                {"x1": 1, "x2": 2, "y1": 2, "y2": 1, "setting": "svd"},
                {"x1": 1, "x2": 2, "y1": 3, "y2": 4, "setting": "qr"},
            ]
        )

        fig, axis = plt.subplots(1, 1)
        _plot_property("x1", "y1", axis, data, log_scale=True, title="")
        self.assertEqual(len(axis.lines), 2)
        x, y = axis.lines[0].get_xydata().T
        self.assertListEqual(list(x), [1])
        self.assertListEqual(list(y), [2])
        x, y = axis.lines[1].get_xydata().T
        self.assertListEqual(list(x), [1])
        self.assertListEqual(list(y), [3])
        fig, axis = plt.subplots(1, 1)
        _plot_property("x2", "y2", axis, data, split_compress=False)
        self.assertEqual(len(axis.lines), 1)
        x, y = axis.lines[0].get_xydata().T
        self.assertListEqual(list(x), [2])
        self.assertListEqual(list(y), [1])
