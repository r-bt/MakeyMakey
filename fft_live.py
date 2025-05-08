import argparse
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from src.radar import Radar
from src.xwr.radar_config import RadarConfig
import cv2

config = None


def update_frame(msg):
    frame = msg.get("data", None)
    if frame is None or config is None:
        return

    range_res = config["range_res"]

    # Average across all chirps
    avg_chirps = np.mean(frame, axis=0)

    # Take the first receiver
    signal = avg_chirps[:, 0]

    # Take the FFT
    fft_result = np.fft(signal)
    fft_result = np.fft.fftshift(fft_result)
    fft_magnitude = np.abs(fft_result)

    # Get the distances
    dists = np.arange(fft_magnitude.shape[0]) * range_res

    # Normalize and visualize
    mag_norm = np.clip(fft_magnitude / np.max(fft_magnitude), 0, 1)
    img = (mag_norm * 255).astype(np.uint8)
    img = np.expand_dims(img, axis=0)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    img_color = cv2.resize(img_color, (1200, 800))

    cv2.imshow("Range FFT - RX0 (0-2m)", img_color)
    cv2.waitKey(1)


def main():
    global config
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")

    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the .lua file used in mmWaveStudio",
    )

    args = parser.parse_args()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()

    # Initialize the radar
    radar = Radar(args.cfg, cb=update_frame)
