import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
import pandas as pd
from src.xwr.dsp import reshape_frame
from src.xwr.radar_config import RadarConfig
import json
import time
from src.dsp import subtract_background


def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()

    c = 3e8  # speed of light - m/s
    SAMPLE_RATE = config["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = config["chirp_slope"]  # frequency slope in Hz (/s)
    SAMPLES_PER_CHIRP = config["n_samples"]  # adc number of samples per chirp

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(0)
    dist_plot.resize(600, 600)
    dist_plot.show()

    # Read the saved data file
    data = np.load(args.data)["data"]

    for frame in data:
        frame = subtract_background(frame)

        # Get the fft of the data
        signal = np.mean(frame, axis=0)

        fft_result = fft(signal, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Threshold the result
        threshold = 600
        fft_result[np.abs(fft_result) < threshold] = 0

        # Plot the data
        dist_plot.update(
            fft_meters[: SAMPLES_PER_CHIRP // 2],
            np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        )

        app.processEvents()

        time.sleep(0.01)


if __name__ == "__main__":
    main()
