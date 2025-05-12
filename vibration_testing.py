import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from src.xwr.radar_config import RadarConfig
import cv2
import matplotlib.pyplot as plt


alpha = 0.6  # decay factor for running average
background = None  # initialize


def subtract_background(current_frame):
    global background
    if background is None:
        background = current_frame.copy()
        return np.zeros_like(current_frame)
    background = alpha * background + (1 - alpha) * current_frame
    return current_frame - background.astype(current_frame.dtype)


def identify_vibrations(heatmap, fft_meters, threshold=1000, max_distance=0.5):
    """
    Takes a heatmap, groups data into clusters and returns the strongest vibrations and their distances

    Args:
        heatmap (np.ndarray): The heatmap to process (vibration_freq_bins, range_bins)
        fft_meters (np.ndarray): The range bins in meters
        threshold (int): The threshold to use for identifying vibrations

    Returns:
        objects (list): A list of dictionaries containing object distance range and all frequencies where greater than threshold
    """
    # Find the indices where the heatmap exceeds the threshold
    indices = np.where(heatmap > threshold)

    locs = zip(indices[0], indices[1])  # (vib_freq, distance)

    # Cluster the locs based on their distances to each other
    clusters = []
    for loc in locs:
        if len(clusters) == 0:
            clusters.append([loc])
        else:
            for cluster in clusters:
                loc_dist = fft_meters[loc[1]]
                cluster_dist = fft_meters[cluster[0][1]]

                if np.abs(loc_dist - cluster_dist) < max_distance:
                    cluster.append(loc)
                    break
            else:
                clusters.append([loc])

    # Calculate the average distance of each cluster
    objects = []
    for cluster in clusters:
        min_dis = np.min([loc[1] for loc in cluster])
        max_dis = np.max([loc[1] for loc in cluster])

        frequencies = [(loc[0], heatmap[loc]) for loc in cluster]

        objects.append(
            {
                "min_distance": min_dis,
                "max_distance": max_dis,
                "frequencies": frequencies,
            }
        )

    return objects


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
    CHIRP_RATE = config["chirp_sampling_rate"]

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    # dist_plot = DistancePlot(0)
    # dist_plot.resize(600, 600)
    # dist_plot.show()

    # Read the saved data file
    data = np.load(args.data)["data"]

    processed_frames = []

    for frame in data:
        # First average across the receivers
        frame = np.mean(frame, axis=2)

        # # First apply a hanning window
        window = np.hanning(SAMPLES_PER_CHIRP)
        frame *= window[None, :]  # apply along samples axis

        frame = np.mean(frame, axis=0)

        # Apply background subtraction
        frame = subtract_background(frame)

        # signal = np.mean(frame, axis=0)

        fft_result = fft(frame, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Only care about the first half of the fft result
        fft_result = fft_result[: SAMPLES_PER_CHIRP // 2]
        fft_freqs = fft_freqs[: SAMPLES_PER_CHIRP // 2]
        fft_meters = fft_meters[: SAMPLES_PER_CHIRP // 2]

        processed_frames.append(fft_result)

    processed_frames = np.array(processed_frames)

    # Split processed_frames into chunks of size 128
    chunk_size = 128
    num_chunks = len(processed_frames) // chunk_size
    processed_frames = np.array_split(processed_frames, num_chunks)

    for chunk in processed_frames:
        heatmap = []

        for range_bin in range(chunk.shape[1]):
            time_series = chunk[:, range_bin]  # shape: (n_chirps,)
            f, t, stft_matrix = stft(time_series, fs=CHIRP_RATE, nperseg=64, noverlap=8)
            magnitude = np.abs(stft_matrix).mean(axis=1)  # avg across time windows
            heatmap.append(magnitude)

        heatmap = np.array(heatmap).T  # shape: (vibration_freq_bins, range_bins)

        # Apply a threshold to the heatmap
        threshold = 100
        heatmap = np.where(heatmap > threshold, heatmap, 0)

        objects = identify_vibrations(
            heatmap, fft_meters, threshold=100, max_distance=0.25
        )

        # Use OpenCV to display the heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (800, 600))

        # Draw lines on the min_distance and max_distance
        for obj in objects:
            min_distance = obj["min_distance"]
            max_distance = obj["max_distance"]

            # Draw a box around the min and max distance
            cv2.rectangle(
                heatmap,
                (int(min_distance), 10),
                (int(max_distance), heatmap.shape[0] - 10),
                np.random.randint(0, 255, 3).tolist(),
                2,
            )

        cv2.imshow("Vibration Intensity Heatmap", heatmap)
        cv2.setWindowTitle("Vibration Intensity Heatmap", "Vibration Intensity Heatmap")

        # print(objects)

        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
