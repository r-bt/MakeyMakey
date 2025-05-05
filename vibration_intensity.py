import numpy as np
import argparse
from src.dsp import reshape_frame
from src.iq_plot import IQPlot, VibrationIntensityPlot
from src.xwr.radar_config import RadarConfig
from PyQt5 import QtWidgets
import sys
import pandas as pd
import json

min_freq = 20  # Minimum frequency in Hz
max_freq = 45  # Maximum frequency in Hz


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

    # Window should be number of sampling points associated with (period of frequency) / 2
    period = 1 / freq

    # window_size = int(chirp_sample_rate * period / 2)
    window_size = int(chirp_sample_rate * period / 2)

    # Create a moving average filter
    kernel = np.ones(window_size) / window_size

    # Apply separately to real and imaginary parts
    real_filtered = np.apply_along_axis(
        lambda m: np.convolve(m.real, kernel, mode="same"), axis=0, arr=frame
    )
    imag_filtered = np.apply_along_axis(
        lambda m: np.convolve(m.imag, kernel, mode="same"), axis=0, arr=frame
    )

    pad = window_size // 2
    trimmed = real_filtered[pad:-pad] + 1j * imag_filtered[pad:-pad]

    return trimmed


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

    # Apply separately to real and imaginary parts
    real_filtered = np.apply_along_axis(
        lambda m: np.convolve(m.real, kernel, mode="same"), axis=0, arr=frame
    )
    imag_filtered = np.apply_along_axis(
        lambda m: np.convolve(m.imag, kernel, mode="same"), axis=0, arr=frame
    )

    # Subtract the filtered data from the original data
    trimmed = frame - (real_filtered + 1j * imag_filtered)

    # Pad the data to match the original shape
    pad = window_size // 2
    trimmed = trimmed[pad:-pad]
    return trimmed


def calculate_vibration_intensity(frame, freq, chirp_sample_rate):
    """
    Calculate the Vibration Intensity (VI) for the given FMCW radar frame.

    Args:
    - frame (np.ndarray): Input radar data with shape (n_chirps, samples_per_chirp, n_receivers).
    - freq (float): The frequency we're interested in.
    - chirp_sample_rate (float): Chirp sample rate (samples per second).

    Returns:
    - np.ndarray: Vibration intensity for each frequency.
    """

    period = 1 / freq  # Period of the frequency
    n_chirps_in_period = int(chirp_sample_rate * period)  # Number of chirps in period
    n_chirps_in_half_period = int(
        chirp_sample_rate * (period / 2)
    )  # Number of chirps in half period

    n_chirps, n_samples, n_receivers = frame.shape

    # Create result arrays to store D_f and D_n for each frequency
    D_f_values = []
    D_n_values = []

    # Iterate over each frequency (the 3rd dimension of the frame)
    for r in range(n_receivers):
        D_f_all = []
        D_n_all = []

        # Iterate over chirps in the valid range, from k0 to k0 + f_a
        for k0 in range(n_chirps - n_chirps_in_period):
            D_f_tmp = []
            D_n_tmp = []

            for k in range(k0, k0 + n_chirps_in_period):

                if k + n_chirps_in_period >= n_chirps:
                    break

                D_f_tmp.append(
                    np.abs(frame[k + n_chirps_in_half_period, :, r] - frame[k, :, r])
                )

                D_n_tmp.append(
                    np.abs(frame[k + n_chirps_in_period, :, r] - frame[k0, :, r])
                )

            D_f_all.append(np.max(D_f_tmp, axis=0))
            D_n_all.append(np.mean(D_n_tmp, axis=0))

        print("D_f_all shape: ", np.array(D_f_all).shape)

        D_f_values.append(np.mean(D_f_all, axis=0))
        D_n_values.append(np.mean(D_n_all, axis=0))

    # Calculate the Vibration Intensity (VI) for each frequency
    VI = np.array(D_f_values) / np.array(D_n_values)

    return VI


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

    iq_plot_raw = IQPlot()
    iq_plot_raw.resize(600, 600)
    iq_plot_raw.show()

    vib_plot = VibrationIntensityPlot()
    vib_plot.resize(600, 600)
    vib_plot.show()

    # Read in the frames
    df = pd.read_csv(args.data, chunksize=1)

    for chunk in df:
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        reshaped_frame = reshape_frame(
            data,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        # Remove random noise
        processed = moving_average(
            reshaped_frame, max_freq, config["chirp_sampling_rate"]
        )

        # Remove baseline drift
        processed = baseline_drift_elimination(
            processed, max_freq, config["chirp_sampling_rate"]
        )

        intensities = []

        for i in range(min_freq, max_freq):
            # Calculate the vibration intensity
            VI = calculate_vibration_intensity(
                processed, i, config["chirp_sampling_rate"]
            )

            # Append the result to the list
            intensities.append(VI)

        # Convert the list to a numpy array
        intensities = np.array(intensities)

        vib_plot.update(intensities)
        iq_plot.update(processed)
        iq_plot_raw.update(reshaped_frame)

        app.processEvents()

    # # Apply moving average to the data
    # chirp_sample_rate = config["chirp_sampling_rate"]
    # moving_avg_frames = [
    #     moving_average(frame, max_freq, chirp_sample_rate) for frame in frames
    # ]

    # print("Shape of moving average frames: ", np.array(moving_avg_frames).shape)
