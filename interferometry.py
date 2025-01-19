from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np

from plotting.plotter import Plotter
from utils import gaussian_model


class Interfermetry:
    def __init__(self, processed_datasets: tuple[tuple[list[float],list[float]]]):
        self._processed_datasets = processed_datasets

    def w_fringes_in_terms_of_visibilities(self):
        w_rad_list = self.narrow_the_search_space(plot=False, find_peaks=False, w_rad_result=True)
        visibilities_and_powers = self.compute_visibility(print_results=False)

        Plotter().plot_w_against_visibility(w_rad_list=w_rad_list, visibilities_and_powers=visibilities_and_powers)

    def narrow_the_search_space(self, plot: bool = True, find_peaks: bool = True, w_rad_result: bool = False):
        power_min_p_avg_deg_datasets = self.convert_to_deg(False)

        power_min_p_avg_deg_datasets_narrowed = []
        peaks_list = []
        w_rad_list = []
        for i in range(len(power_min_p_avg_deg_datasets)):
            deg, power = power_min_p_avg_deg_datasets[i]

            if i==0:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=7.75, max_deg=9, distance=305
                )
            elif i==1:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=5.75, max_deg=7, distance=360
                )
            elif i==2:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=1, max_deg=5, distance=700
                )
            elif i==3:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=6.4, max_deg=7.45, distance=250
                )
            elif i==4:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=6, max_deg=6.55, distance=125
                )
            elif i==5:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=5.12, max_deg=6.5, distance=360
                )
            elif i==6:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=6, max_deg=7.5, distance=360
                )
            elif i==7:
                narrow_deg, narrow_power, peaks, w_rad = self._find_W_fringe(
                    deg, power, min_deg=2.5, max_deg=10.5, distance=1800
                )

            w_rad_list.append(w_rad)
            peaks_list.append(peaks)
            power_min_p_avg_deg_datasets_narrowed.append(tuple((narrow_deg, narrow_power)))

        if plot:
            if find_peaks:
                Plotter().plot_narrow_data_with_peaks(power_min_p_avg_deg_datasets_narrowed, tuple(peaks_list))
            else:
                Plotter().plot_raw_data(power_min_p_avg_deg_datasets_narrowed, x_axis_label="Deg (°)")

        if w_rad_result:
            return w_rad_list

        return power_min_p_avg_deg_datasets_narrowed

    def convert_to_deg(self, plot: bool = True):
        power_min_p_avg_datasets = self.remove_p_avg(False)

        power_min_p_avg_deg_datasets = []
        for i in range(len(power_min_p_avg_datasets)):
            time, power = power_min_p_avg_datasets[i]
            
            deg = (np.array(time) * 360) / 86400

            power_min_p_avg_deg_datasets.append(tuple((deg.tolist(), power)))

        if plot:
            Plotter().plot_raw_data(power_min_p_avg_deg_datasets, x_axis_label="Deg (°)")

        return power_min_p_avg_deg_datasets

    def remove_p_avg(self, plot: bool = True):
        visibilities_and_powers = self.compute_visibility(print_results=False)
        all_gaussians = self._fit_gaussians()
        
        power_min_p_avg_datasets = []
        for i in range(len(visibilities_and_powers)):
            time, power = self._processed_datasets[i]

            popt_big, pcov_big, popt_small, pcov_small = all_gaussians[i]
            
            big_gaussian = gaussian_model(time, *popt_big)
            small_gaussian = gaussian_model(time, *popt_small)

            p_avg = (np.array(big_gaussian) + np.array(small_gaussian)) / 2

            power_min_p_avg = np.array(power) - p_avg

            power_min_p_avg_datasets.append(tuple((time, power_min_p_avg)))

        if plot:
            Plotter().plot_raw_data(power_min_p_avg_datasets)

        return power_min_p_avg_datasets


    def compute_visibility(self, print_results: bool = True):
        all_pmin_pmax = self._compute_pmax_pmin()

        dataset = ["A", "B", "C", "D", "E", "F", "G", "H"]

        visibilities_and_powers = []
        for i in range(len(all_pmin_pmax)):    
            p_max, p_max_error, p_min, p_min_error = all_pmin_pmax[i]
            visibility = (p_max-p_min) / (p_max+p_min)
            visibility_error = (
                (
                    ((2*p_min) / (p_max+p_min)**2) * p_max_error
                )**2
                +
                (
                    ((2*p_max) / (p_max+p_min)**2) * p_min_error
                )**2
                )**0.5

            p_avg = (p_max + p_min) / 2
            p_avg_error = (
                (0.5 * p_max_error) ** 2
                +
                (0.5 * p_min_error) ** 2
            )**0.5
            
            max_ = "{max}"
            min_ = "{min}"
            avg_ = "{avg}"
            
            if print_results:
                print(f"========= Baseline {dataset[i]} =========")
                print("-------------- Power --------------")
                print(f"P_{max_} = {p_max:e} \\pm {p_max_error:e} W")
                print(f"P_{min_} = {p_min:e} \\pm {p_min_error:e} W")
                print(f"P_{avg_} = {p_avg:e} \\pm {p_avg_error:e} W")
                print("----------- Visibility -----------")
                print(f"|V(B_\\lambda)| = {visibility:e} \\pm {visibility_error:e}")

            visibilities_and_powers.append(
                tuple(
                    (
                        p_max,
                        p_max_error,
                        p_min,
                        p_min_error,
                        p_avg,
                        p_avg_error,
                        visibility,
                        visibility_error
                    )
                )
            )
        return tuple(visibilities_and_powers)
    

    def plot_peaks_and_troughs(self):
        """
        Plots the peaks and the troughs of the data.
        """
        Plotter().plot_processed_data_with_peaks_and_troughs(self._processed_datasets, self._find_peaks_and_troughs())

    def plot_gaussians(self):
        all_gaussians = self._fit_gaussians()
        peaks_and_troughs = self._find_peaks_and_troughs()
        Plotter().plot_processed_data_with_gaussian(self._processed_datasets, peaks_and_troughs, all_gaussians)

    def _find_W_fringe(self, deg: list[float], power: list[float], min_deg: float, max_deg: float, distance: float = 40):
        deg_arr = np.array(deg)

        min_deg_index = np.where(deg_arr > min_deg)[0][0]
        max_deg_index = np.where(deg_arr > max_deg)[0][0]

        narrow_deg = deg[min_deg_index:max_deg_index]
        narrow_power = power[min_deg_index:max_deg_index]

        peaks_indecies = find_peaks(narrow_power, distance=distance)[0]

        peaks_deg = np.array(narrow_deg)[peaks_indecies]
        peaks_power = np.array(narrow_power)[peaks_indecies]

        peaks = tuple((peaks_deg.tolist(), peaks_power.tolist()))

        if len(peaks_deg) == 2:
            w_deg = np.abs(peaks_deg[0] - peaks_deg[-1])
            w_rad = np.deg2rad(w_deg)
        else:
            raise ValueError("Found more than two peaks while trying to compute W_fringe!")

        return narrow_deg, narrow_power, peaks, w_rad

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
            elif i == 7:
                p0_guess_big = [np.max(peak_powers), peak_times[len(peak_times)//2], 1000, np.min(peak_powers)]
                p0_guess_small = [np.max(trough_powers), 1000, 1000, np.min(trough_powers)]
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
        popt_big, pcov_big = curve_fit(gaussian_model, peak_times, peak_powers, p0=p0_guess_big)#, bounds=(0,1e6))
        
        popt_small, pcov_small = curve_fit(gaussian_model, trough_times, trough_powers, p0=p0_guess_small)#, bounds=(0,1e6))

        return popt_big, pcov_big, popt_small, pcov_small
    
    def _compute_pmax_pmin(self) -> tuple[tuple[float,float,float,float]]:
        all_gaussians = self._fit_gaussians()

        all_pmin_pmax = []
        for i in range(len(all_gaussians)):
            popt_big, pcov_big, popt_small, pcov_small = all_gaussians[i]
            p_max = popt_big[0] + popt_big[-1]
            p_max_error = (pcov_big[-1,-1]**2 + pcov_big[0,0]**2)**0.5

            p_min = popt_small[0] + popt_small[-1]
            p_min_error = (pcov_small[-1,-1]**2 + pcov_small[0,0]**2)**0.5
            all_pmin_pmax.append(tuple((p_max, p_max_error, p_min, p_min_error)))

        return tuple(all_pmin_pmax)
    

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
    