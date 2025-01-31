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
        bassel_model_donut: Callable[..., Any],
        sinc_model: Callable[..., Any],
        p0_bessel: list[float],
        p0_bessel_donut: list[float],
        p0_sinc: list[float],
        min_point: list[float, float],
    ):
        """
        Plots fitted Bessel, Donut Bessel, and Sinc functions against visibility data.

        :param baseline: The baseline data points.
        :type baseline: list[float]
        :param visibility: The visibility data corresponding to the baseline.
        :type visibility: list[float]
        :param bassel_model: The Bessel model function used for fitting.
        :type bassel_model: Callable[..., Any]
        :param bassel_model_donut: The Donut Bessel model function used for fitting.
        :type bassel_model_donut: Callable[..., Any]
        :param sinc_model: The Sinc model function used for fitting.
        :type sinc_model: Callable[..., Any]
        :param p0_bessel: Initial parameters for the Bessel model fitting.
        :type p0_bessel: list[float]
        :param p0_bessel_donut: Initial parameters for the Donut Bessel model fitting.
        :type p0_bessel_donut: list[float]
        :param p0_sinc: Initial parameters for the Sinc model fitting.
        :type p0_sinc: list[float]
        :param min_point: The minimum point (first root) for the Sinc function.
        :type min_point: list[float, float]
        """
        linspace_x = np.linspace(0, np.array(baseline).max(), 1000)

        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        # Bessel Plot
        axs[0].scatter(baseline, visibility, label="Data", color="b")
        axs[0].plot(
            linspace_x,
            bassel_model(linspace_x, *p0_bessel),
            label=f"Fitted Bessel\na = {p0_bessel[0]:.2f}\nb = {p0_bessel[1]:.2f}",
            color="r",
            linestyle="-",
        )
        axs[0].set_title("Bassel function fit")
        axs[0].set_xlabel("Baseline (λ)")
        axs[0].set_ylabel("Visibility")
        axs[0].grid(True)
        axs[0].legend()

        # Bessel Donnut plot
        axs[1].scatter(baseline, visibility, label="Data", color="b")
        axs[1].plot(
            linspace_x,
            bassel_model_donut(linspace_x, *p0_bessel_donut),
            label=(
                f"Fitted Donut Bessel\nBig:\na = {p0_bessel_donut[0]:.2f}\n"
                f"b = {p0_bessel_donut[1]:.2f}\n"
                f"Small:\na = {p0_bessel_donut[2]:.2f}\n"
                f"b = {p0_bessel_donut[3]:.2f}"
            ),
            color="r",
            linestyle="-",
        )
        axs[1].set_title("Donut Bassel function fit")
        axs[1].set_xlabel("Baseline (λ)")
        axs[1].set_ylabel("Visibility")
        axs[1].grid(True)
        axs[1].legend(loc="right")

        # Sinc plot
        axs[2].scatter(baseline, visibility, label="Data", color="b")
        axs[2].plot(
            linspace_x,
            sinc_model(linspace_x, *p0_sinc),
            label=f"Fitted Sinc\na = {p0_sinc[0]:.2f}\nb = {p0_sinc[1]:.2f}",
            color="r",
            linestyle="-",
        )
        axs[2].axvline(
            min_point[0], label=f"First root: B={min_point[0]:.2f}", color="g"
        )
        # axs[2].scatter(min_point[0], min_point[1], label=f"First root: B={min_point[0]:.2f}", color='g')
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
        """
        Plots visibility against baseline with error bars.

        :param baseline: The baseline data points.
        :type baseline: np.ndarray
        :param visibilities: The visibility data corresponding to the baseline.
        :type visibilities: list[float]
        :param visibilities_error: The error in the visibility data.
        :type visibilities_error: list[float]
        """
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
        """
        Plots processed data with Gaussian fits overlaid.

        :param datasets: Tuple containing datasets of time and power values.
        :type datasets: tuple[tuple[list[float], list[float]]]
        :param peaks_and_troughs: Tuple containing peak and trough data for the datasets.
        :type peaks_and_troughs: tuple[tuple[list[float], list[float]]]
        :param all_gaussians: Tuple containing Gaussian fit parameters for all datasets.
        :type all_gaussians: tuple[tuple[np.ndarray, np.ndarray]]
        """
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
        """
        Plots processed data with peaks and troughs identified.

        :param datasets: Tuple containing datasets of time and power values.
        :type datasets: tuple[tuple[list[float], list[float]]]
        :param peaks_and_troughs: Tuple containing peak and trough data for the datasets.
        :type peaks_and_troughs: tuple[tuple[list[float], list[float]]]
        """
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
        """
        Plots narrow data with peak markers.

        :param datasets: Tuple containing datasets of time and power values.
        :type datasets: tuple[tuple[list[float], list[float]]]
        :param peaks: Tuple containing the peak positions for each dataset.
        :type peaks: tuple[list[float]]
        """
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
        y_axis_label: str = "Power (dBm)",
    ) -> None:
        """
        Makes a 2x4 figure of all the raw data from the datasets.

        :param datasets: the datastets:
        "A", "B", "C", "D", "E", "F", "G", and "H"
        :type datasets: tuple[tuple[list[float],list[float]]]
        """
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        self._plot_data(
            datasets_names,
            datasets,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
        )

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
        """
        Helper method for plotting data with optional overlays for peaks, troughs, and Gaussian fits.

        :param datasets_names: The names of the datasets to be plotted.
        :type datasets_names: list[str]
        :param datasets: Tuple containing datasets of time and power values.
        :type datasets: tuple[tuple[list[float], list[float]]]
        :param processed: Whether the data is processed (i.e., should peaks and troughs be plotted).
        :type processed: bool
        :param peaks_and_troughs: Tuple containing peak and trough data for the datasets (optional).
        :type peaks_and_troughs: tuple[tuple[list[float], list[float]]] | None
        :param peaks_narrow: Tuple containing narrow peak positions for each dataset (optional).
        :type peaks_narrow: tuple[tuple[list[float], list[float]]] | None
        :param y_axis_label: The label for the y-axis. Defaults to "Power (dBm)".
        :type y_axis_label: str
        :param x_axis_label: The label for the x-axis. Defaults to "Time (s)".
        :type x_axis_label: str
        :param gaussian: Whether to overlay Gaussian fits on the data.
        :type gaussian: bool
        :param all_gaussians: Tuple containing Gaussian fit parameters for all datasets (optional).
        :type all_gaussians: tuple[tuple[np.ndarray, np.ndarray]] | None
        :param peaks: Whether to overlay peak markers on the data.
        :type peaks: bool
        """
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
