import argparse
import numpy as np
from src.xwr.radar_config import RadarConfig
import pandas as pd
import json
from src.dsp import reshape_frame
from PyQt5 import QtWidgets
from src.iq_plot import IQPlot, VibrationIntensityPlot
import sys
import time

MIN_FREQ = 20  # Minimum frequency in Hz
MAX_FREQ = 45  # Maximum frequency in Hz


def random_noise_elimination(frame, freq, chirp_sample_rate):
    """
    Remove the random noise from the data.

    From HomeOSD: The random noise can be removed by using the moving average of a window with the length of the period
    of frequency / 2.

    Args:
        frame (np.ndarray): The input fft array (n_chirps_per_frame, samples_per_chirp, n_receivers).
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

    # pad = window_size // 2
    # trimmed = real_filtered[pad:-pad] + 1j * imag_filtered[pad:-pad]

    return real_filtered + 1j * imag_filtered


def remove_baseline_drift(frame, freq, chirp_sample_rate):
    """
    Remove the baseline drift from the data.

    From HomeOSD: An approximate drift component can be obtained by using the moving average of a window with the length of the period
    of frequency.

    Args:
        frame (np.ndarray): The input fft array (n_chirps_per_frame, samples_per_chirp, n_receivers).
        freq (float): The frequency we're interested in.
        chirp_sample_rate (float): The chirp sample rate in Hz.
    """
    # Window should be number of sampling points associated with (period of frequency) / 2
    period = 1 / freq

    # window_size = int(chirp_sample_rate * period / 2)
    window_size = int(chirp_sample_rate * period)

    # Create a moving average filter
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

    # Pad the trimmed data to remove the edges
    # pad = window_size // 2
    # trimmed = real_filtered[pad:-pad] + 1j * imag_filtered[pad:-pad]

    return frame - (real_filtered + 1j * imag_filtered)


def vibration_intensity(frame, freq, chirp_sample_rate):
    period = 1 / freq
    chirps_in_period = int(chirp_sample_rate * period)
    chirps_in_half_period = chirps_in_period // 2

    n_chirps, samples_per_chirp, n_receivers = frame.shape
    max_k0 = n_chirps - chirps_in_period * 2  # ensures k + chirps_in_period < n_chirps

    if max_k0 <= 0:
        return np.zeros(n_receivers)

    vi = np.zeros(n_receivers)

    for i in range(n_receivers):
        k0 = np.arange(max_k0)[:, None]  # (n_windows, 1)
        k = k0 + np.arange(chirps_in_period)  # (n_windows, chirps_in_period)
        k_half = k + chirps_in_half_period
        k_full = k + chirps_in_period

        ref = frame[k, :, i]  # (n_windows, chirps_in_period, samples)
        half = frame[k_half, :, i]
        full = frame[k_full, :, i]

        df = np.max(np.abs(half - ref), axis=2)  # (n_windows, chirps_in_period)
        dn = np.mean(np.abs(full - ref), axis=2)

        vi[i] = np.max(df) / np.mean(dn)

    return vi


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

    vib_plot = VibrationIntensityPlot()
    vib_plot.resize(600, 600)
    vib_plot.show()

    # Read in the frames
    df = pd.read_csv(args.data, chunksize=1)

    for chunk in df:
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        # Reshape the data into a 3D array (n_chirps_per_frame, samples_per_chirp, n_receivers) of IQ samples
        reshaped_frame = reshape_frame(
            data,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        # Apply an FFT to the frame across the samples
        fft_data = np.fft.fft(reshaped_frame, axis=1)
        fft_data = np.fft.fftshift(fft_data, axes=1)

        vibration_intensity_data = []

        freqs = np.array(range(MIN_FREQ, MAX_FREQ))

        # for i in freqs:
        # Remove the random noise
        processed_fft_data = random_noise_elimination(
            fft_data, MAX_FREQ, config["chirp_sampling_rate"]
        )

        # # Remove the baseline drift
        # processed_fft_data = remove_baseline_drift(
        #     processed_fft_data, i, config["chirp_sampling_rate"]
        # )

        # # Calculate the vibration intensity
        # vib_intensity = vibration_intensity(
        #     processed_fft_data, i, config["chirp_sampling_rate"]
        # )

        # vibration_intensity_data.append((i, vib_intensity[0]))

        # vib_plot.update(vibration_intensity_data)
        real_samples = processed_fft_data.real.flatten()
        imag_samples = processed_fft_data.imag.flatten()

        # Update the GUI with the new data
        data = np.column_stack((real_samples, imag_samples))

        iq_plot.update(data)

        app.processEvents()
