import matplotlib.pyplot as plt


class Plotter:
    def plot_raw_data(self, datasets: tuple[tuple[list[float],list[float]]]) -> None:
        """
        Makes a 2x4 figure of all the raw data from the datasets.

        :param datasets: the datastets:
        "A", "B", "C", "D", "E", "F", "G", and "H"
        :type datasets: tuple[tuple[list[float],list[float]]]
        """
        datasets_names = ["A", "B", "C", "D", "E", "F", "G", "H"]

        # Create a 4x2 fig
        _, axes = plt.subplots(2, 4, figsize=(15, 10))

        axes = axes.flatten()

        # Plot on each subplot
        for i in range(len(datasets_names)):
            time, power = datasets[i]
            axes[i].plot(time, power, label="Power")
            axes[i].set_title(f"Baseline {datasets_names[i]}")
            axes[i].legend()

        plt.tight_layout()

        plt.savefig("raw_data_plot.png")
        plt.show()