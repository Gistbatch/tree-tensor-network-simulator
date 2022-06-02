"""Visualization of experiments."""
import json
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _plot_exp(file: str, x_axis: str, properties: List[str], fix_mps: bool = True):
    plt.style.use("ggplot")
    with open(f"./data/{file}.json", "r", encoding="utf-8") as in_file:
        data = json.load(in_file)
    data = pd.DataFrame(data)
    mps = data[data["setting"] == "mps"]
    if fix_mps:
        mps = mps.iloc[0]
    ttn = data[data["setting"] != "mps"]

    _, axes = plt.subplots(len(properties), 1)
    if len(properties) == 1:
        axes = [axes]
    for axis, prop in zip(axes, properties):
        if fix_mps:
            axis.axhline(mps[prop], c="y", ls="-", label="MPS")
        else:
            axis.plot(mps[str.lower(x_axis)], mps[prop], "-", label="MPS")
        _plot_property(
            str.lower(x_axis),
            prop,
            axis,
            ttn,
            y_axis="$t_{" + prop + "}\ [s]$",
            split_compress=fix_mps,
        )
        axis.semilogy(base=10)
    axes[-1].set_xlabel(x_axis)
    axes[0].legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.savefig(f"./data/{file}.pdf", format="pdf")


def _plot_property(
    x_values: str,
    y_values: str,
    axis,
    data: pd.DataFrame,
    **kwargs,
) -> None:

    axis.set_title(kwargs.get("title", ""))
    axis.set_xlabel(kwargs.get("x_axis", ""))
    axis.set_ylabel(kwargs.get("y_axis", ""))
    svd = data[data["setting"] == "svd"]
    axis.plot(svd[x_values], svd[y_values], "-", label="TTN SVD")
    if kwargs.get("split_compress", True):
        qr_data = data[data["setting"] == "qr"]
        axis.plot(qr_data[x_values], qr_data[y_values], "-", label="TTN QR")
    if kwargs.get("log_scale", False):
        axis.semilogy(base=2)


def _plot_dry_runs(file: str):
    plt.style.use("ggplot")
    with open(f"./data/{file}.json", "r", encoding="utf-8") as in_file:
        data = json.load(in_file)
    data = pd.DataFrame(data)
    default = data[data["setting"] == "default"]
    t1_data = data[data["setting"] == "t1"]
    bounded = data[data["setting"] == "bounded"]
    minimized_sum = data[data["setting"] == "minimized_sum"]
    mps = data[data["setting"] == "mps"]
    _, (ax0, ax1) = plt.subplots(2, 1)

    ax0.plot(default["qubits"], default["sum_prod_dims"], "-", label="TTN default")
    ax0.plot(
        default["qubits"], [2 ** x for x in default["qubits"]], "-", label="Statevector"
    )
    ax0.plot(
        t1_data["qubits"], t1_data["sum_prod_dims"], "-", label="TTN mixed setting"
    )
    ax0.plot(
        bounded["qubits"],
        bounded["sum_prod_dims"],
        "-",
        label="TTN bounded cluster size",
    )
    ax0.plot(
        minimized_sum["qubits"],
        minimized_sum["sum_prod_dims"],
        "-",
        label="TTN minimized bond sum",
    )
    ax0.plot(mps["qubits"], mps["sum_prod_dims"], "-", label="MPS")
    ax0.set_ylabel(r"$M_{\mathrm{TN}} $")
    ax0.semilogy(base=2)
    ax0.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=3,
        mode="expand",
        borderaxespad=0.0,
        fontsize=8,
    )

    ax1.set_xlabel("#Qubits")
    ax1.plot(default["qubits"], default["max_dim"], "-", label="TTN default")
    ax1.plot(t1_data["qubits"], t1_data["max_dim"], "-", label="TTN mixed setting")
    ax1.plot(
        bounded["qubits"], bounded["max_dim"], "-", label="TTN bounded cluster size"
    )
    ax1.plot(
        minimized_sum["qubits"],
        minimized_sum["max_dim"],
        "-",
        label="TTN minimized bond sum",
    )
    ax1.plot(mps["qubits"], mps["max_dim"], "-", label="MPS")
    ax1.set_ylabel(r"$D_{\max}$")

    ax1.semilogy(base=2)
    plt.tight_layout()
    plt.savefig(f"./data/{file}.pdf", format="pdf")


def create_plots() -> None:
    """
    Wrapper function to create plots for experiments.
    """
    properties = ["find", "apply"]
    _plot_exp("structure_experiment1", "Clusters", properties)
    _plot_exp("sycamore_experiment1", "Clusters", properties)
    _plot_exp("structure_experiment3", "Qubits", ["apply"], fix_mps=False)
    _plot_dry_runs("structure_experiment2")
    _plot_dry_runs("sycamore_experiment2")

if __name__ == "__main__":
    create_plots()
