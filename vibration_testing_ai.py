import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from src.xwr.radar_config import RadarConfig
from src.dsp import subtract_background, identify_vibrations

# Other mills at 2.35, 3.9

# OTHERMILL_DISTANCES = [1.3]
OTHERMILL_DISTANCES = [2.35, 3.9]
DISTANCE_THRESHOLD = 0.1

alpha = 0.6  # decay factor for running average
background = None  # initialize


def subtract_background_1(current_frame):
    global background
    if background is None:
        background = current_frame.copy()
        return np.zeros_like(current_frame)
    background = alpha * background + (1 - alpha) * current_frame
    return current_frame - background.astype(current_frame.dtype)


def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    name = args.data.split("/")[-1].split(".")[0]

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

    all_objs = []

    for chunk in processed_frames:
        heatmap = []

        for range_bin in range(chunk.shape[1]):
            time_series = chunk[:, range_bin]  # shape: (n_chirps,)
            f, t, stft_matrix = stft(time_series, fs=CHIRP_RATE, nperseg=64, noverlap=8)
            magnitude = np.abs(stft_matrix).mean(axis=1)  # avg across time windows
            heatmap.append(magnitude)

        heatmap = np.array(heatmap).T  # shape: (vibration_freq_bins, range_bins)

        # Apply a threshold to the heatmap
        # threshold = 100
        # heatmap = np.where(heatmap > threshold, heatmap, 0)

        objects = identify_vibrations(
            heatmap,
            fft_meters,
            othermills=OTHERMILL_DISTANCES,
            max_distance=DISTANCE_THRESHOLD,
        )

        all_objs.extend(objects)

    np.save(
        f"training/{name}.npy",
        all_objs,
        allow_pickle=True,
    )

    # for obj in objects:
    #     slice_heatmap = cv2.normalize(obj, None, 0, 255, cv2.NORM_MINMAX)
    #     slice_heatmap = np.uint8(slice_heatmap)
    #     slice_heatmap = cv2.applyColorMap(slice_heatmap, cv2.COLORMAP_JET)

    #     slice_heatmap = cv2.resize(slice_heatmap, (800, 600))

    #     cv2.imshow("Vibration Intensity Heatmap", slice_heatmap)
    #     cv2.setWindowTitle(
    #         "Vibration Intensity Heatmap", "Vibration Intensity Heatmap"
    #     )

    #     while True:
    #         if cv2.waitKey(1) & 0xFF == ord("q"):
    #             break

    # for obj in objects:
    #     print("Mean distance:", np.mean([obj["min_distance"], obj["max_distance"]]))
    #     if (
    #         np.abs(
    #             OTHERMILL_DISTANCES[0]
    #             - np.mean([obj["min_distance"], obj["max_distance"]])
    #         )
    #         < DISTANCE_THRESHOLD
    #     ):
    #         detections += 1
    #         # print("Detected object at distance:", obj["min_distance"], "m")
    #         # print("Frequencies:", obj["frequencies"])

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


if __name__ == "__main__":
    main()
