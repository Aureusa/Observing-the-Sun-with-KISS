import matplotlib.pyplot as plt
import numpy as np

from utils import gaussian_model


class Plotter:
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
            y_axis_label: str = "Power (dBm)",
            x_axis_label: str = "Time (s)",
            gaussian: bool = False,
            all_gaussians: tuple[tuple[np.ndarray,np.ndarray]]|None = None
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