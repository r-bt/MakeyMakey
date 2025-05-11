import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq


def sliding_window_average(frame, window_size):
    """
    Compute the moving average of the data across frames

    Args:
        frame (np.ndarray): The input data array (n_chirps_per_frame, n_fft_bins)
        window_size (int): The size of the moving average window.
    """
    kernel = np.ones(window_size) / window_size
    filtered = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=frame
    )
    pad = window_size // 2
    return filtered[pad:-pad]


def background_subtraction(frame):
    after_subtraction = np.zeros_like(frame)
    for i in range(1, frame.shape[0]):
        after_subtraction[i - 1] = frame[i] - frame[i - 1]

    return after_subtraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Initalize the radar
    radar = Radar(args.cfg)

    params = radar.params

    c = 3e8  # speed of light - m/s
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(params["range_res"])
    dist_plot.resize(600, 600)
    dist_plot.show()

    def update_frame(msg):
        frame = msg.get("data", None)
        if frame is None:
            return
        frame = sliding_window_average(frame, 10)
        frame = background_subtraction(frame)

        # Get the fft of the data
        signal = np.mean(frame, axis=0)

        fft_result = fft(signal, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Plot the data
        dist_plot.update(
            fft_meters[: SAMPLES_PER_CHIRP // 2],
            np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        )

        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
