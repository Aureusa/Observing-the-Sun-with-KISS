from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np

from plotting.plotter import Plotter
from utils import gaussian_model


class Interfermetry:
    def __init__(self, processed_datasets: tuple[tuple[list[float],list[float]]]):
        self._processed_datasets = processed_datasets

    def plot_peaks_and_troughs(self):
        """
        Plots the peaks and the troughs of the data.
        """
        Plotter().plot_processed_data_with_peaks_and_troughs(self._processed_datasets, self._find_peaks_and_troughs())

    def plot_gaussians(self):
        all_gaussians = self._fit_gaussians()
        peaks_and_troughs = self._find_peaks_and_troughs()
        Plotter().plot_processed_data_with_gaussian(self._processed_datasets, peaks_and_troughs, all_gaussians)

    def _fit_gaussians(self) -> tuple[tuple[np.ndarray,np.ndarray]]:
        all_peaks_and_troughs = self._find_peaks_and_troughs()

        all_gaussians = []
        for i in range(len(all_peaks_and_troughs)):
            peak_times, peak_powers, trough_times, trough_powers = all_peaks_and_troughs[i]

            if i == 0:
                # Outlier removal
                outlier_peak_index = np.argmax(np.array(peak_powers))
                del peak_powers[outlier_peak_index]
                del peak_times[outlier_peak_index]

                p0_guess_big = [np.max(peak_powers), peak_times[len(peak_times)//2], 1000, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers)-np.min(trough_powers), peak_times[len(peak_times)//2], 600, 5.75e-8]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            elif i == 1:
                p0_guess_big = [np.max(peak_powers), peak_times[len(peak_times)//2], 1000, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers)-np.min(trough_powers), 1500, 500, 5.5e-8]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            elif i == 2:
                p0_guess_big = [1.4e-7, 750, 150, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers)-np.min(trough_powers), 1500, 500, 5.5e-8]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            elif i == 5:
                p0_guess_big = [np.max(peak_powers)-np.min(peak_powers), 1250, 300, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers)-np.min(trough_powers), 1500, 500, 5.5e-8]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            elif i == 6:
                p0_guess_big = [np.max(peak_powers)-np.min(peak_powers), 1600, 300, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers)-np.min(trough_powers), 1900, 350, np.min(trough_powers)]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            elif i == 4:
                p0_guess_big = [np.max(peak_powers), peak_times[len(peak_times)//2], 1000, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers), trough_times[int(len(trough_times)//2)], 1000, np.min(trough_powers)]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )
            else:
                p0_guess_big = [np.max(peak_powers), peak_times[len(peak_times)//2], 1000, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers), trough_times[int(len(trough_times)//2)], 1000, np.min(trough_powers)]
                popt_big, pcov_big, popt_small, pcov_small = self._fit_two_gaussians(
                    peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small
                )

            all_gaussians.append(tuple((popt_big, pcov_big, popt_small, pcov_small)))

        return tuple(all_gaussians)
    
    def _fit_two_gaussians(self, peak_times, peak_powers, trough_times, trough_powers, p0_guess_big, p0_guess_small):
        popt_big, pcov_big = curve_fit(gaussian_model, peak_times, peak_powers, p0=p0_guess_big)
        
        popt_small, pcov_small = curve_fit(gaussian_model, trough_times, trough_powers, p0=p0_guess_small)

        return popt_big, pcov_big, popt_small, pcov_small


    def _find_peaks_and_troughs(self) -> tuple[tuple[list[float],list[float]]]:
        """
        Finds the peaks of the datasets.

        :return: the peaks of the datasets
        :rtype: tuple[tuple[list[float],list[float]]]
        """
        all_peaks_and_troughs = []

        for i in range(len(self._processed_datasets)):
            # Unpack the dataset tuple
            time, power = self._processed_datasets[i]
            
            # Find the indecies of troughs and peaks
            if i == 2:
                peaks_indecies, trough_indecies = self._find_indecies(power, 700)
                peaks_indecies = np.append(peaks_indecies,trough_indecies[-1])
                trough_indecies = np.append(trough_indecies,trough_indecies[-2])
            elif i == 7:
                peaks_indecies, trough_indecies = self._find_indecies(power, 1300)
            else:
                peaks_indecies, trough_indecies = self._find_indecies(power, 210)

            # Perform a list comprehansion to extract the
            # times and powers of the peaks
            peak_times, peak_powers = self._list_comprehansion(peaks_indecies, time, power)

            # Perform a list comprehansion to extract the
            # times and powers of the troughs
            trough_times, trough_powers = self._list_comprehansion(trough_indecies, time, power)

            # Append them by creating a tuple for each dataset
            all_peaks_and_troughs.append(tuple((peak_times, peak_powers, trough_times, trough_powers)))

        return tuple(all_peaks_and_troughs)
    
    def _find_indecies(self, power: list[float], distance: int) -> tuple[list[float],list[float]]:
        """
        Finds the indecies of the peaks and troughs.

        :param power: the power to find the peaks and troughs of.
        :type power: list[float]
        :param distance: the distance parmaeter of "find_peaks" func
        from scipy.signal
        :type distance: int
        :return: a tuple containing the indecies of the peaks and troughs
        :rtype: tuple[list[float],list[float]]
        """

        peaks_indecies = find_peaks(power, distance=distance)[0]

        inverted_power_arr = - np.array(power)
        inverted_power = inverted_power_arr.tolist()
        trough_indecies = find_peaks(inverted_power, distance=distance)[0]
        return peaks_indecies, trough_indecies
    
    def _list_comprehansion(self,indecies: list[float], time: list[float], power: list[float]) -> tuple[list[float],list[float]]:
        """
        Performs a list comprehansion to extract datapoint (time, power).

        :param indecies: the indecies
        :type indecies: list[float]
        :param time: the list of times
        :type time: list[float]
        :param power: the list of powers
        :type power: list[float]
        :return: the datapoints
        :rtype: tuple[list[float],list[float]]
        """
        times = []
        powers = []
        for index in indecies:
            time_ = time[index]
            power_ = power[index]

            times.append(time_)
            powers.append(power_)

        return times, powers
    