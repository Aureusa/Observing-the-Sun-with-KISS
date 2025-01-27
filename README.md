
# Observing the Sun with KISS

This repository contains the final project for the "Introduction to Radio Astronomy" course. The project focuses on observing the Sun using KISS (Kapteyn Interferometer for Short-baseline Solar observations)

## Project Structure

The repository is organized as follows:

- `data/`: Contains raw data collected during the observations.
- `graphs_for_project/`: Includes graphs and visualizations generated for the project report.
- `plotting/`: Contains scripts used for plotting data.

Key files in the repository:

- `interferometry.py`: A Python script related to interferometry calculations.
- `main.py`: The main script for processing and analyzing the observational data.
- `utils.py`: Utility functions used across various scripts.

## Requirements

The project is implemented in Python. To run the scripts, ensure you have the following packages installed:

- `numpy`
- `matplotlib`
- `scipy`

You can install the required packages using `pip`:

```bash
pip install numpy matplotlib scipy
```

## Usage

To analyze the observational data, run the `main.py` script:

```bash
python main.py
```

This will process the data in the `data/` directory and generate relevant plots.

## License

While this project is licensed under the "All Rights Reserved" license, TAs are granted permission to view the code exclusively for the purpose of grading this report. No part of this repository may be copied, modified, or distributed without explicit permission from the copyright holder.
