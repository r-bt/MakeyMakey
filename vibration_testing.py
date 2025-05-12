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
import time
from src.dsp import subtract_background

alpha = 0.6  # decay factor for running average
background = None  # initialize

OTHERMILL_DISTANCES = [1.3]
DISTANCE_THRESHOLD = 0.1


def subtract_background_1(current_frame):
    global background
    if background is None:
        background = current_frame.copy()
        return np.zeros_like(current_frame)
    background = alpha * background + (1 - alpha) * current_frame
    return current_frame - background.astype(current_frame.dtype)


def identify_vibrations(heatmap, fft_meters, threshold=1000, max_distance=0.25):
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
        min_dis = np.min([fft_meters[loc[1]] for loc in cluster])
        max_dis = np.max([fft_meters[loc[1]] for loc in cluster])

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
        frame = subtract_background_1(subtract_background(frame))

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

    detections = 0

    for chunk in processed_frames:
        heatmap = []

        for range_bin in range(chunk.shape[1]):
            time_series = chunk[:, range_bin]  # shape: (n_chirps,)
            f, t, stft_matrix = stft(time_series, fs=CHIRP_RATE, nperseg=64, noverlap=8)
            magnitude = np.abs(stft_matrix).mean(axis=1)  # avg across time windows
            heatmap.append(magnitude)

        heatmap = np.array(heatmap).T  # shape: (vibration_freq_bins, range_bins)

        print(np.max(heatmap))

        # Apply a threshold to the heatmap
        threshold = 50
        heatmap = np.where(heatmap > threshold, heatmap, 0)

        objects = identify_vibrations(
            heatmap, fft_meters, threshold=100, max_distance=0.2
        )

        for obj in objects:
            print("Mean distance:", np.mean([obj["min_distance"], obj["max_distance"]]))
            if (
                np.abs(
                    OTHERMILL_DISTANCES[0]
                    - np.mean([obj["min_distance"], obj["max_distance"]])
                )
                < DISTANCE_THRESHOLD
            ):
                detections += 1
                # print("Detected object at distance:", obj["min_distance"], "m")
                # print("Frequencies:", obj["frequencies"])

        # # # Use OpenCV to display the heatmap
        # heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        # heatmap = np.uint8(heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # heatmap = cv2.resize(heatmap, (800, 600))

        # cv2.imshow("Vibration Intensity Heatmap", heatmap)
        # cv2.setWindowTitle("Vibration Intensity Heatmap", "Vibration Intensity Heatmap")

        # while True:
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        # Use matplotlib to display the heatmap
        step = max(1, len(fft_meters) // 10)  # show ~10 ticks
        xticks = np.arange(0, len(fft_meters), step)

        plt.imshow(heatmap, cmap="hot", interpolation="nearest", aspect="auto")
        plt.colorbar()
        plt.title("Vibration Intensity Heatmap")
        # plt.xticks(
        #     ticks=xticks,
        #     labels=[f"{fft_meters[i]:.2f}" for i in xticks],
        #     rotation=45,
        #     ha="right",
        # )

        print("distance", fft_meters[40])

        plt.show()

        time.sleep(1)

        plt.close("all")

    print("Accuracy = ", (detections / len(processed_frames)))


if __name__ == "__main__":
    main()
