import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any

from utils import gaussian_model


class Plotter:
    def plot_fitted_funcs(
        self,
        baseline: list[float],
        visibility: list[float],
        bassel_model: Callable[..., Any],
        basel_model_params: list[float],
        bassel_model_donut: Callable[..., Any],
        basel_model_donut_params: list[float],
        sinc_model: Callable[..., Any],
        sinc_model_params: list[float],
        p0_bessel: list[float],
        p0_bessel_donut: list[float],
        p0_sinc: list[float],
    ):
        linspace_x = np.linspace(0, np.array(baseline).max(), 1000)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        axs[0].scatter(baseline, visibility, label="Data", color="b")
        axs[0].plot(
            linspace_x,
            bassel_model(linspace_x, *basel_model_params),
            label=f"Fitted Bessel\na = {basel_model_params[0]:.2f}\nb = {basel_model_params[1]:.2f}",
            color="g",
            linestyle="--",
        )
        # axs[0].plot(linspace_x, bassel_model(linspace_x, *p0_bessel),
        #             label="Initial guess", color='r', linestyle='-')
        axs[0].set_title("Bassel function fit")
        axs[0].set_xlabel("Baseline (λ)")
        axs[0].set_ylabel("Visibility")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].scatter(baseline, visibility, label="Data", color="b")
        axs[1].plot(
            linspace_x,
            bassel_model_donut(linspace_x, *basel_model_donut_params),
            label=(
                f"Fitted Donut Bessel\nBig:\na = {basel_model_donut_params[0]:.2f}\n"
                f"b = {basel_model_donut_params[1]:.2f}\n"
                f"Small:\na = {basel_model_donut_params[2]:.2f}\n"
                f"b = {basel_model_donut_params[3]:.2f}"),
            color="g",
            linestyle="--",
        )
        # axs[1].plot(linspace_x, bassel_model_donut(linspace_x, *p0_bessel_donut),
        #             label="Initial guess", color='r', linestyle='-')
        axs[1].set_title("Donut Bassel function fit")
        axs[1].set_xlabel("Baseline (λ)")
        axs[1].set_ylabel("Visibility")
        axs[1].grid(True)
        axs[1].legend(loc="right")

        axs[2].scatter(baseline, visibility, label="Data", color="b")
        axs[2].plot(
            linspace_x,
            sinc_model(linspace_x, *sinc_model_params),
            label=f"Fitted Sinc\na = {sinc_model_params[0]:.2f}\nb = {sinc_model_params[1]:.2f}",
            color="g",
            linestyle="--",
        )
        # axs[2].plot(linspace_x, sinc_model(linspace_x, *p0_sinc),
        #             label="Initial guess", color='r', linestyle='-')
        axs[2].set_title("Sinc function fit")
        axs[2].set_xlabel("Baseline (λ)")
        axs[2].set_ylabel("Visibility")
        axs[2].grid(True)
        axs[2].legend()

        fig.tight_layout(pad=5.0)
        plt.show()

    def plot_w_against_visibility(
        self,
        baseline: np.ndarray,
        visibilities: list[float],
        visibilities_error: list[float],
    ):
        visibilities_arr = np.array(visibilities)

        _, ax = plt.subplots()

        # Plot the data with error bars
        ax.errorbar(
            baseline,
            visibilities_arr,
            yerr=visibilities_error,
            fmt="o",
            ecolor="red",
            capsize=5,
            label="Data with Error Bars",
        )

        # Customize the plot
        ax.set_title("Visibility in terms of baseline")
        ax.set_xlabel("Baseline (λ)")
        ax.set_ylabel("Visibility")
        ax.legend()
        ax.grid(True)

        plt.show()

    def plot_processed_data_with_gaussian(
        self,
        datasets: tuple[tuple[list[float], list[float]]],
        peaks_and_troughs: tuple[tuple[list[float], list[float]]],
        all_gaussians: tuple[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        self._plot_data(
            datasets_names,
            datasets,
            processed=True,
            peaks_and_troughs=peaks_and_troughs,
            y_axis_label="Power (W)",
            gaussian=True,
            all_gaussians=all_gaussians,
        )

    def plot_processed_data_with_peaks_and_troughs(
        self,
        datasets: tuple[tuple[list[float], list[float]]],
        peaks_and_troughs: tuple[tuple[list[float], list[float]]],
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        self._plot_data(
            datasets_names,
            datasets,
            processed=True,
            peaks_and_troughs=peaks_and_troughs,
            y_axis_label="Power (W)",
        )

    def plot_narrow_data_with_peaks(
        self,
        datasets: tuple[tuple[list[float], list[float]]],
        peaks: tuple[list[float]],
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        self._plot_data(
            datasets_names=datasets_names,
            datasets=datasets,
            peaks_narrow=peaks,
            x_axis_label="Deg (°)",
            y_axis_label="Power (W)",
            peaks=True,
        )

    def plot_raw_data(
        self,
        datasets: tuple[tuple[list[float], list[float]]],
        x_axis_label: str = "Time (s)",
    ) -> None:
        """
        Makes a 2x4 figure of all the raw data from the datasets.

        :param datasets: the datastets:
        "A", "B", "C", "D", "E", "F", "G", and "H"
        :type datasets: tuple[tuple[list[float],list[float]]]
        """
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        self._plot_data(datasets_names, datasets, x_axis_label=x_axis_label)

    def _plot_data(
        self,
        datasets_names: list[str],
        datasets: tuple[tuple[list[float], list[float]]],
        processed: bool = False,
        peaks_and_troughs: tuple[tuple[list[float], list[float]]] | None = None,
        peaks_narrow: tuple[tuple[list[float], list[float]]] | None = None,
        y_axis_label: str = "Power (dBm)",
        x_axis_label: str = "Time (s)",
        gaussian: bool = False,
        all_gaussians: tuple[tuple[np.ndarray, np.ndarray]] | None = None,
        peaks: bool = False,
    ) -> None:
        # Create a 4x2 fig
        fig, axes = plt.subplots(2, 4, figsize=(20, 15))

        axes = axes.flatten()

        # Plot on each subplot
        for i in range(len(datasets_names)):
            time, power = datasets[i]
            axes[i].plot(time, power, label="Power", color="blue")

            if processed:
                (peak_times, peak_powers, trough_times, trough_powers) = (
                    peaks_and_troughs[i]
                )

                axes[i].scatter(
                    peak_times, peak_powers, label="Peaks", color="red", marker="o"
                )
                axes[i].scatter(
                    trough_times,
                    trough_powers,
                    label="Troughs",
                    color="yellow",
                    marker="o",
                )

            if peaks:
                peaks_degs, _ = peaks_narrow[i]
                for peaks in peaks_degs:
                    axes[i].axvline(x=peaks, color="red", ls="--")

            if gaussian:
                popt_big, _, popt_small, _ = all_gaussians[i]
                y_gauss_big = gaussian_model(np.array(time), *popt_big)
                y_gauss_small = gaussian_model(np.array(time), *popt_small)

                axes[i].plot(
                    time, y_gauss_big, label="Max power gaussian", color="green"
                )
                axes[i].plot(
                    time, y_gauss_small, label="Min power gaussian", color="pink"
                )

            axes[i].set_title(f"Baseline {datasets_names[i]}")

            if gaussian is False:
                axes[i].legend()

        fig.text(0.5, 0.04, x_axis_label, ha="center")
        fig.text(0.04, 0.5, y_axis_label, va="center", rotation="vertical")

        plt.show()
