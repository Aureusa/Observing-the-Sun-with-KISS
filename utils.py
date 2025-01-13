import os
import numpy as np

DATA_DIR = os.path.join(os.getcwd(), "data")

def get_data(folder: str = "dataset_1") -> tuple[tuple[list[float],list[float]]]:
    """
    Gets all the data from a given folder

    :param folder: the folder name, defaults to "dataset_1"
    :type folder: str, optional
    :return: a tuple containing all datasets
    :rtype: _type_
    """
    target_folder = os.path.join(DATA_DIR, folder)

    dataset = ["A", "B", "C", "D", "E", "F", "G", "H"]

    all_datasets = []

    for set in dataset:
        all_powers = []
        all_times = []
        with open(os.path.join(target_folder, f"baseline_{set}.txt"), "r") as file:
            for line in file:
                line_ = line.strip()
                split_line = line_.split()
                time = split_line[0]
                power = split_line[1]

                # This try-except block servers to remove
                # the first line of the dataset
                try:
                    all_powers.append(float(power))
                    all_times.append(float(time))
                except ValueError:
                    continue

        all_datasets.append(tuple((all_times,all_powers)))

    return tuple(all_datasets)

def preprocess_data(datasets: tuple[tuple[list[float],list[float]]]) -> tuple[tuple[list[float],list[float]]]:
    processed_data = []
    for i in range(len(datasets)):
        time, power = datasets[i]

        # Convert power from [dBm] -> [W]
        exponent = np.array(power) * 10 ** (-1)
        power_mili_watt = 10 ** (exponent)
        power_watt = power_mili_watt * 0.001

        processed_data.append(tuple((time, power_watt.tolist())))
    return tuple(processed_data)

def gaussian_model(x, A, mean, sigma, offset):
    gaussian = np.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return A * gaussian + offset
