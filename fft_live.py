import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq, fftshift


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
        global count
        global background
        frame = msg.get("data", None)
        if frame is None:
            return
        frame = background_subtraction(frame)
        # Get the fft of the data
        signal = np.mean(frame, axis=0)

        # signal = signal - background

        fft_result = fft(signal, axis=0)
        # Get the doppler shift of the data by taking a second fft
        # doppler_result = fft(fft_result, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # # Second fft for doppler shift
        # doppler_fft = fftshift(fft(fft_result))
        # doppler_freqs = fft_freqs

        # # A separate plot to test with
        # plt.plot(doppler_freqs, np.abs(doppler_fft))
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Magnitude')
        # plt.show

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
