import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
from src.doppler_plot import DopplerPlot
import sys
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from collections import deque

def background_subtraction(frame):
    after_subtraction = np.zeros_like(frame)
    for i in range(1, frame.shape[0]):
        after_subtraction[i - 1] = frame[i] - frame[i - 1]

    return after_subtraction

def sliding_window(frame, buffer, window_size):
    buffer.append(frame)
    if len(buffer) == window_size:
        window_data = np.stack(buffer, axis=0)
        processed_frame = np.mean(window_data, axis=0)
        return processed_frame
    return frame

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
    
    # doppler_plot = DopplerPlot(params["range_res"])
    # doppler_plot.resize(600, 600)
    # doppler_plot.show()

    WINDOW_SIZE = 8
    window_buffer = deque(maxlen=WINDOW_SIZE)

    def update_frame(msg):
        global count
        global background
        frame = msg.get("data", None)
        params = msg.get("params", None)
        if frame is None:
            return

        processed_frame = sliding_window(frame, window_buffer, WINDOW_SIZE)


        # frame = reshape_frame(
        #     frame,
        #     params["n_chirps"],
        #     params["n_samples"],
        #     params["n_rx"],
        # )

        # frame = background_subtraction(frame)
        # Get the fft of the data
        signal = np.mean(processed_frame, axis=1)

        n_samples = SAMPLES_PER_CHIRP

        if (len(signal) < n_samples):
            signal_padded = np.zeros(n_samples)
            signal_padded[:len(signal)] = signalsignal = signal_padded
        else:
            signal = signal[:n_samples]

        # signal = signal - background

        fft_result = fft(signal)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        half_n = n_samples//2
        half_fft = np.abs(fft_result[:half_n])
        half_meters = fft_meters[:half_n]

        plot_y = half_fft.reshape(-1, 1)
        plot_x = half_meters

        if (len(plot_x) != len(plot_y)):
            min_len = min(len(plot_x), len(plot_y))
            plot_x = plot_x[:min_len]
            plot_y = plot_y[:min_len]

        # Second fft for doppler shift
        doppler_fft = fftshift(fft(fft_result))
        doppler_data = np.abs(doppler_fft[:half_n]).reshape(-1, 1)

        if len(plot_x) != len(doppler_data):
            doppler_data = doppler_data[:len(plot_x)]


        # Plot the data
        dist_plot.update(
            plot_x,
            np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        )

        doppler_plot.update(
            plot_x, 
            doppler_data,
        )
        app.processEvents()

    # Initialize the radar

    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
