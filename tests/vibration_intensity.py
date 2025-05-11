import numpy as np
import argparse
from src.xwr.dsp import reshape_frame
from src.iq_plot import IQPlot, VibrationIntensityPlot
from src.xwr.radar_config import RadarConfig
from PyQt5 import QtWidgets
import sys
import pandas as pd
import json
import matplotlib.pyplot as plt
import time
import concurrent.futures

min_freq = 15  # Minimum frequency in Hz
max_freq = 120  # Maximum frequency in Hz


def moving_average(frame, freq, chirp_sample_rate):
    """
    Compute the moving average of the data across frames

    Used to remove random noise from the data.

    From the HomeOSD paper, we want the window size to be the number of sampling points associated with the period of frequency / 2.

    Args:
        frame (np.ndarray): The input data array (n_chirps_per_frame, samples_per_chirp, n_receivers).
        freq (float): The frequency we're interested in.
        chirp_sample_rate (float): The chirp sample rate in Hz.
    """

    period = 1 / freq
    window_size = int(chirp_sample_rate * period / 2)

    if window_size < 1:
        return frame  # Avoid empty kernel

    kernel = np.ones(window_size) / window_size

    # Apply moving average directly to complex data
    filtered = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=frame
    )

    pad = window_size // 2
    return filtered[pad:-pad]


def baseline_drift_elimination(frame, freq, chirp_sample_rate):
    """
    Remove the baseline drift from the data.

    From HomeOSD: An approximate drift component can be obtained by using the moving average of a window with the length of the period

    Args:
        frame (np.ndarray): The input data array (n_chirps_per_frame, samples_per_chirp, n_receivers).
        freq (float): The frequency we're interested in.
    """

    period = 1 / freq

    window_size = int(chirp_sample_rate * period)

    kernel = np.ones(window_size) / window_size

    # Apply moving average directly to complex data
    filtered = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=frame
    )

    # Subtract the filtered data from the original data
    trimmed = frame - filtered

    # Pad the data to match the original shape
    pad = window_size // 2
    trimmed = trimmed[pad:-pad]
    return trimmed


def vibration_intensity(frame, freq, chirp_sample_rate):
    period = 1 / freq
    chirps_in_period = int(chirp_sample_rate * period)
    chirps_in_half_period = chirps_in_period // 2

    n_chirps, _, _ = frame.shape

    dfs = []
    dns = []

    for k in range(n_chirps - chirps_in_period):
        dfs.append(
            np.max(
                np.abs(frame[k + chirps_in_half_period, :, :] - frame[k, :, :]), axis=0
            )
        )

        dns.append(
            np.mean(np.abs(frame[k + chirps_in_period, :, :] - frame[k, :, :]), axis=0)
        )

    df = np.mean(dfs, axis=0)
    dn = np.mean(dns, axis=0)

    # Calculate the vibration intensity
    return df / dn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()

    n_chirps_per_frame = config["n_chirps"]
    n_receivers = config["n_rx"]
    samples_per_chirp = config["n_samples"]

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    iq_plot = IQPlot()
    iq_plot.resize(600, 600)
    iq_plot.show()

    # iq_plot_raw = IQPlot()
    # iq_plot_raw.resize(600, 600)
    # iq_plot_raw.show()

    vib_plot = VibrationIntensityPlot()
    vib_plot.resize(600, 600)
    vib_plot.show()

    # Read in the frames
    df = pd.read_csv(args.data, chunksize=1)

    prev_frame = None

    for chunk in df:
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        reshaped_frame = reshape_frame(
            data,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        vibration_intensity_data = []

        def process_frequency(i):
            # Remove random noise
            processed = moving_average(reshaped_frame, i, config["chirp_sampling_rate"])

            # Calculate the vibration intensity
            vib = vibration_intensity(processed, i, config["chirp_sampling_rate"])
            return i, vib[0]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            vibration_intensity_data = list(
                executor.map(
                    lambda i: process_frequency(i),
                    range(min_freq, max_freq),
                )
            )

        vib_plot.update(vibration_intensity_data)
        iq_plot.update(reshaped_frame[:, :, 0].reshape(-1))
        app.processEvents()
