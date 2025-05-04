import numpy as np
import argparse
from src.dsp import reshape_frame
from src.iq_plot import IQPlot
from PyQt5 import QtWidgets
import sys
import pandas as pd
import json

min_freq = 20  # Minimum frequency in Hz
max_freq = 45  # Maximum frequency in Hz

# TODO: Get these from the config
n_receivers = 4
samples_per_chirp = 128
n_chirps_per_frame = 128


def moving_average(data, freq):
    """
    Compute the moving average of the data across frames

    Args:
        data (np.ndarray): The input data array (frames, n_chirps_per_frame, samples_per_chirp, n_receivers).
        window_size (int): The size of the moving average window.
    """

    # Window should be number of sampling points associated with period of frequency / 2
    period = 1 / freq

    window_size = 

    cumsum = np.cumsum(data, axis=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1 :] / window_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    iq_plot = IQPlot()
    iq_plot.resize(600, 600)
    iq_plot.show()

    # Read in the frames
    df = pd.read_csv(args.data, chunksize=1)

    frames = []

    for chunk in df:
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        reshaped_frame = reshape_frame(
            data,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        frames.append(reshaped_frame)

    for freq in range(min_freq, max_freq + 1):
        
