import matplotlib.pyplot as plt
import numpy as np

from utils import gaussian_model


class Plotter:
    def plot_w_against_visibility(self, w_rad_list: list[float], visibilities_and_powers: tuple):
        # Calculate the baseline
        baseline = 1/np.array(w_rad_list)

        visibilities = []
        visibilities_error = []

        # Unpack visibilitiies and powers tuple
        for i in range(len(visibilities_and_powers)):
            (
                _,
                _,
                _,
                _,
                _,
                _,
                visibility,
                visibility_error
            ) = visibilities_and_powers[i]

            visibilities.append(visibility)
            visibilities_error.append(visibility_error)

        visibilities_arr = np.array(visibilities)
        
        normalized_visibilities = visibilities_arr# / visibilities_arr.sum()
        normalized_visibility_errors = np.array(visibilities_error)# / visibilities_arr.sum()

        _, ax = plt.subplots()

        # Plot the data with error bars
        ax.errorbar(
            baseline,
            normalized_visibilities,
            yerr=normalized_visibility_errors,
            fmt='o',
            ecolor='red',
            capsize=5,
            label='Data with Error Bars'
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
            datasets: tuple[tuple[list[float],list[float]]],
            peaks_and_troughs: tuple[tuple[list[float],list[float]]],
            all_gaussians: tuple[tuple[np.ndarray,np.ndarray]]
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        self._plot_data(
            datasets_names,
            datasets,
            processed=True,
            peaks_and_troughs=peaks_and_troughs,
            y_axis_label="Power (W)",
            gaussian=True,
            all_gaussians=all_gaussians
        )


    def plot_processed_data_with_peaks_and_troughs(
            self,
            datasets: tuple[tuple[list[float],list[float]]],
            peaks_and_troughs: tuple[tuple[list[float],list[float]]]
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        self._plot_data(datasets_names, datasets, processed=True, peaks_and_troughs=peaks_and_troughs, y_axis_label="Power (W)")

    def plot_narrow_data_with_peaks(
            self,
            datasets: tuple[tuple[list[float],list[float]]],
            peaks: tuple[list[float]]
    ) -> None:
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
        
        self._plot_data(
            datasets_names=datasets_names,
            datasets=datasets,
            peaks_narrow=peaks,
            x_axis_label = "Deg (°)",
            y_axis_label="Power (W)",
            peaks=True
        )

    def plot_raw_data(self, datasets: tuple[tuple[list[float],list[float]]], x_axis_label: str = "Time (s)") -> None:
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
            datasets: tuple[tuple[list[float],list[float]]],
            processed: bool = False,
            peaks_and_troughs: tuple[tuple[list[float],list[float]]]|None = None,
            peaks_narrow: tuple[tuple[list[float],list[float]]]|None = None,
            y_axis_label: str = "Power (dBm)",
            x_axis_label: str = "Time (s)",
            gaussian: bool = False,
            all_gaussians: tuple[tuple[np.ndarray,np.ndarray]]|None = None,
            peaks: bool = False
    ) -> None:
        # Create a 4x2 fig
        fig, axes = plt.subplots(2, 4, figsize=(20, 15))

        axes = axes.flatten()

        # Plot on each subplot
        for i in range(len(datasets_names)):
            time, power = datasets[i]
            axes[i].plot(time, power, label="Power", color="blue")

            if processed:
                (
                    peak_times,
                    peak_powers,
                    trough_times,
                    trough_powers
                ) = peaks_and_troughs[i]

                axes[i].scatter(peak_times, peak_powers, label="Peaks", color="red", marker="o")
                axes[i].scatter(trough_times, trough_powers, label="Troughs", color="yellow", marker="o")

            if peaks:
                peaks_degs, _ = peaks_narrow[i]
                for peaks in peaks_degs:
                    axes[i].axvline(x = peaks, color = "red", ls="--")

            if gaussian:
                popt_big, _, popt_small, _ = all_gaussians[i]
                y_gauss_big = gaussian_model(np.array(time), *popt_big)
                y_gauss_small = gaussian_model(np.array(time), *popt_small)

                axes[i].plot(time, y_gauss_big, label="Max power gaussian", color="green")
                axes[i].plot(time, y_gauss_small, label="Min power gaussian", color="pink")


            axes[i].set_title(f"Baseline {datasets_names[i]}")

            if gaussian is False:
                axes[i].legend()

        fig.text(0.5, 0.04, x_axis_label, ha="center")
        fig.text(0.04, 0.5, y_axis_label, va="center", rotation='vertical')

        #plt.tight_layout()

        plt.show()