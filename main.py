from utils import get_data, preprocess_data
from plotting.plotter import Plotter
from interferometry import Interfermetry


# Unpack the data
all_data_sets = get_data()

# Preprocess the data
all_data_sets_processed = preprocess_data(all_data_sets)

# Instantiate an interermoter
interfermoeter = Interfermetry(all_data_sets_processed)

def main():
    # Plot the raw data
    Plotter().plot_raw_data(all_data_sets)

    # Plot the raw data with peaks and troughs
    interfermoeter.plot_peaks_and_troughs()

    # Fit gaussians through the peaks and troughs and plot them
    interfermoeter.plot_gaussians()

    # Substract the average power from the signal
    interfermoeter.remove_p_avg()

    # Converts time to degrees
    interfermoeter.convert_to_deg()

    # "Zoom in" so that only two peaks are considered
    interfermoeter.narrow_the_search_space()

    # Compute the W_fringe for each data set and plots the
    # visibilities in terms of the baselines
    interfermoeter.w_fringes_in_terms_of_visibilities()

    # Fit a Bessel, Bessel Donut, and sinc func through
    # the visibilities in terms of baseline
    interfermoeter.fit_funcs()

if __name__ == "__main__":
    main()
