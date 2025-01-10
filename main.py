from utils import get_data, preprocess_data
from plotting.plotter import Plotter

# (
#     dataset_A,
#     dataset_B,
#     dataset_C,
#     dataset_D,
#     dataset_E,
#     dataset_F,
#     dataset_G,
#     dataset_H
# ) = get_data()

all_data_sets = get_data()

all_data_sets_processed = preprocess_data(all_data_sets)

Plotter.plot_raw_data(all_data_sets)
