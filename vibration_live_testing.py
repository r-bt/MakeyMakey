import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import pandas as pd
from src.xwr.dsp import reshape_frame
from src.xwr.radar_config import RadarConfig
import time
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


processed_frames = []


def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar
    radar = Radar(args.cfg)

    params = radar.params

    c = 3e8  # speed of light - m/s
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    CHIRP_RATE = params["chirp_sampling_rate"]

    def update_frame(msg):
        global processed_frames
        # First get the frame
        frame = msg.get("data", None)
        if frame is None:
            return

        # Average across the receivers
        frame = np.mean(frame, axis=2)

        # Apply a hanning window
        window = np.hanning(SAMPLES_PER_CHIRP)
        frame *= window[None, :]  # apply along samples axis

        # Average across the chirps
        frame = np.mean(frame, axis=0)

        # Apply background subtraction
        frame = subtract_background(frame)

        fft_result = fft(frame, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Only care about the first half of the fft result
        fft_result = fft_result[: SAMPLES_PER_CHIRP // 2]
        fft_freqs = fft_freqs[: SAMPLES_PER_CHIRP // 2]
        fft_meters = fft_meters[: SAMPLES_PER_CHIRP // 2]

        processed_frames.append(fft_result)

        if len(processed_frames) > 50:  # About 5s worth of data
            p_frames = np.array(processed_frames)

            heatmap = []

            for range_bin in range(p_frames.shape[1]):
                time_series = p_frames[:, range_bin]  # shape: (n_chirps,)
                f, t, stft_matrix = stft(
                    time_series, fs=CHIRP_RATE, nperseg=25, noverlap=16
                )
                magnitude = np.abs(stft_matrix).mean(axis=1)  # avg across time windows
                heatmap.append(magnitude)

            heatmap = np.array(heatmap).T  # shape: (vibration_freq_bins, range_bins)

            # Apply a threshold to the heatmap
            threshold = 1000
            heatmap = np.where(heatmap > threshold, heatmap, 0)

            # Use cv2 to display the heatmap
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = np.uint8(heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imshow("Heatmap", heatmap)
            cv2.waitKey(1)

            processed_frames = []

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
