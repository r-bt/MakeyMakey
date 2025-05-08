import argparse
from src.xwr.radar_config import RadarConfig
import pandas as pd
import json
import numpy as np
from src.dsp import reshape_frame
from scipy.signal import butter, filtfilt
import cv2
from scipy.signal import stft


def compute_stft_vibration(data, target_range_bin=None, fs=1.0):
    """
    data: np.array of shape (n_chirps, n_samples, n_rx)
    fs: sampling frequency along chirp axis (slow time)
    target_range_bin: int or None → if None, process all bins

    Returns:
        stft_results: list of STFT spectrograms (one per RX antenna)
    """
    # 1. Range FFT
    range_fft = np.fft.fft(data, axis=1)[
        :, : data.shape[1] // 2, :
    ]  # (n_chirps, n_range_bins, n_rx)

    # 2. Optional: pick a range bin (you can iterate over bins too)
    if target_range_bin is None:
        target_range_bin = np.argmax(
            np.sum(np.abs(range_fft), axis=(0, 2))
        )  # strongest reflector

    # 3. Extract slow-time signal per RX
    stft_results = []
    for rx in range(data.shape[2]):
        signal = range_fft[:, target_range_bin, rx]
        f, t, Zxx = stft(signal, fs=fs, window="hann", nperseg=32, noverlap=28)
        stft_results.append((f, t, np.abs(Zxx)))

    return stft_results  # [(frequencies, times, magnitude_spectrogram), ...]


def normalize_and_color(data, max_val=0, cmap=None):
    if max_val <= 0:
        max_val = np.max(data)

    img = (data / max_val * 255).astype(np.uint8)
    if cmap:
        img = cv2.applyColorMap(img, cmap)
    return img


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

    # Read in the frames
    df = pd.read_csv(args.data, chunksize=1)

    for chunk in df:
        # Read in the data
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        # Reshape the data into a 3D array (n_chirps_per_frame, samples_per_chirp, n_receivers) of IQ samples
        reshaped_frame = reshape_frame(
            data, n_chirps_per_frame, samples_per_chirp, n_receivers
        )

        res = compute_stft_vibration(
            reshaped_frame, target_range_bin=None, fs=1.0
        )

        # # Display the data
        # img = normalize_and_color(
        #     np.abs(vibration_spectrum.T), max_val=0, cmap=cv2.COLORMAP_JET
        # )

        # img = cv2.resize(img, (800, 600))

        # # Display the image
        # cv2.imshow("FFT Data", img)
        # cv2.waitKey(1)  # Allow OpenCV to process the window events
