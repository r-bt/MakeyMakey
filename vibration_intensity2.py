import argparse
import numpy as np
from src.xwr.radar_config import RadarConfig
import pandas as pd
import json
from src.dsp import reshape_frame
from PyQt5 import QtWidgets
from src.iq_plot import IQPlot, VibrationIntensityPlot, VibrationIntensityHeatmap
import sys
import time
import concurrent.futures

MIN_FREQ = 25  # Minimum frequency in Hz
MAX_FREQ = 80  # Maximum frequency in Hz


def random_noise_elimination(frame, freq, chirp_sample_rate):
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

    return frame - (real_filtered + 1j * imag_filtered)


def vibration_intensity(frame, freq, chirp_sample_rate):
    """
    Computes the vibration intensity of the data.

    Args:
        frame (np.ndarray): The input fft data array (n_chirps_per_frame, n_freq_bins, n_receivers).
        freq (float): The frequency we're interested in.
        chirp_sample_rate (float): The chirp sample rate in Hz.

    Returns:
        np.ndarray: The vibration intensity of the data (n_freq_bins, n_receivers).
    """
    period = 1 / freq
    chirps_in_period = int(chirp_sample_rate * period)
    chirps_in_half_period = chirps_in_period // 2

    n_chirps = frame.shape[0]
    max_k0 = n_chirps - chirps_in_period

    # Precompute index arrays once
    offset = np.arange(chirps_in_period)
    k0 = np.arange(max_k0)
    k = k0[:, None] + offset
    k_half = k + chirps_in_half_period
    k_full = k + chirps_in_period

    # Filter valid indices
    valid_mask = (k_full < n_chirps).all(axis=1)
    k, k_half, k_full = k[valid_mask], k_half[valid_mask], k_full[valid_mask]

    # Reshape once for efficient fancy indexing
    f_k = frame[k]  # shape: (valid_k, chirps_in_period, freq_bins, receivers)
    f_k_half = frame[k_half]
    f_k_full = frame[k_full]

    # Compute differences in-place
    df_all = np.abs(f_k_half - f_k)
    dn_all = np.abs(f_k_full - f_k)

    df = np.mean(np.max(df_all, axis=1), axis=0)
    dn = np.mean(np.mean(dn_all, axis=1), axis=0)

    return df / dn


def estimate_object_distances(vib_data, vib_threshold, dist_threshold, range_res):
    """
    Estimate the distances of objects based on the vibration intensity data.

    Args:
        vib_data (np.ndarray): The vibration intensity data (n_freqs, n_distances)
        vib_threshold (float): The threshold for vibration intensity.
        dist_threshold (float): The threshold for distance estimation.
        range_res (float): The range resolution of the radar.

    Returns:
        objects (np.ndarray): Minimum and maximum distances of objects in class
    """
    # Find the indices where the vibration intensity exceeds the threshold
    indices = np.where(vib_data > vib_threshold)

    locs = zip(indices[0], indices[1])

    # Cluster the locs based on their distances to each other
    clusters = []

    for loc in locs:
        found = False
        for cluster in clusters:
            loc_dist = loc[1] * range_res
            cluster_dist = cluster[0][1] * range_res

            if loc_dist - cluster_dist < dist_threshold:
                cluster.append(loc)
                found = True
                break

        if not found:
            clusters.append([loc])

    objects = []
    for cluster in clusters:
        min_dist = min([loc[1] for loc in cluster])
        max_dist = max([loc[1] for loc in cluster])
        objects.append((min_dist, max_dist))

    return objects


def feature_extraction(vib_data, obj, k=5):
    """
    Returns top k freqs and VI values for a given object inside a distance range

    Args:
        vib_data (np.ndarray): The vibration intensity data (n_freqs, n_distances)
        object (tuple): The object to extract features from (min_dist, max_dist).
    """

    min_dist, max_dist = obj

    # Get the indices of the distances that are inside the range
    data = vib_data[:, min_dist:max_dist]

    # Get max values at each freq
    max_values = np.max(data, axis=1)

    # Get the indices of the top k values
    top_k_indices = np.argsort(max_values)[-k:]
    top_k_values = max_values[top_k_indices]

    return zip(top_k_indices, top_k_values)


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

    vib_plot = VibrationIntensityHeatmap(
        start_freq=MIN_FREQ, range_res=config["range_res"]
    )
    vib_plot.resize(600, 600)
    vib_plot.show()

    # Read in the frames
    df = pd.read_csv(args.data, skiprows=range(1, 200))

    frames = np.array([json.loads(row) for row in df["data"]], dtype=np.int16)

    res = []

    for frame in frames:
        start_time = time.time()

        reshaped_frame = reshape_frame(
            frame,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        vibration_intensity_data = []

        print(1 / config["t_sweep"])

        def process_frequency(i):
            # Remove random noise
            processed = random_noise_elimination(
                reshaped_frame, i, 1 / config["t_sweep"]
            )

            # Apply an FFT to the frame across the samples
            fft_data = np.fft.fft(processed, axis=1)
            fft_data = np.fft.fftshift(fft_data, axes=1)

            # We only care about close objects for testing (helps with speed)
            fft_data = fft_data[:, : samples_per_chirp // 4, :]

            # Calculate the vibration intensity
            vib = vibration_intensity(fft_data, i, 1 / config["t_sweep"])
            return np.max(vib, axis=1)

        t = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            vibration_intensity_data = list(
                executor.map(
                    lambda i: process_frequency(i),
                    range(MIN_FREQ, MAX_FREQ),
                )
            )

        vib_data = np.array(vibration_intensity_data)

        objects = estimate_object_distances(
            vib_data, vib_threshold=3, dist_threshold=0.5, range_res=config["range_res"]
        )

        found = False

        for obj in objects:
            if obj[0] * config["range_res"] > 0 and obj[1] * config["range_res"] < 2:
                res.append(0)
                found = True
                break

        if not found:
            res.append(1)

        # features = [list(feature_extraction(vib_data, obj)) for obj in objects]

        # for fe

        # print(features)

        # vib_plot.update(vib_data)

        # real_samples = reshaped_frame[:, :, 0].real.flatten()
        # imag_samples = reshaped_frame[:, :, 0].imag.flatten()

        # # Update the GUI with the new data
        # data = np.column_stack((real_samples, imag_samples))

        # t = time.time()
        # iq_plot.update(data)

        app.processEvents()

    accuracy = sum(res) / len(res)
    print(f"Accuracy: {accuracy:.2f}")
