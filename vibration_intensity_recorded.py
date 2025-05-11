import argparse
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
from src.vibration_intensity_plot import VibrationIntensityHeatmap
import sys
from scipy.fft import fft, fftfreq
from src.xwr.radar_config import RadarConfig
import time
import concurrent.futures
import numpy as np

MIN_FREQ = 80  # Minimum frequency in Hz
MAX_FREQ = 140  # Maximum frequency in Hz


def random_noise_elimination(frame, freq, chirp_sample_rate):
    """
    Compute the moving average of the data across frames

    Used to remove random noise from the data.

    From the HomeOSD paper, we want the window size to be the number of sampling points associated with the period of frequency of interest / 2.

    Args:
        frame (np.ndarray): The input data array (n_chirps_per_frame, n_fft_bins)
        freq (float): The frequency we're interested in.
        chirp_sample_rate (float): The chirp sample rate in Hz.
    """
    period = 1 / freq
    window_size = int(chirp_sample_rate * period / 2)

    if window_size < 1:
        return frame  # Avoid empty kernel

    kernel = np.ones(window_size) / window_size

    # Apply moving average directly to complex data across the chirps
    filtered = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=frame
    )

    pad = window_size // 2
    return filtered[pad:-pad]


def vibration_intensity(frame, freq, chirp_sample_rate):
    period = 1 / freq
    chirps_in_period = int(chirp_sample_rate * period)
    chirps_in_half_period = chirps_in_period // 2

    n_chirps, _ = frame.shape

    dfs = []
    dns = []

    for k0 in range(0, n_chirps - chirps_in_period):
        df_tmp = []
        dn_tmp = []

        for k in range(k0, k0 + chirps_in_period):
            if k + chirps_in_period >= n_chirps:
                break

            df_tmp.append(np.abs(frame[k + chirps_in_half_period] - frame[k]))
            dn_tmp.append(np.abs(frame[k + chirps_in_period] - frame[k]))

        dfs.append(np.max(df_tmp, axis=0))
        dns.append(np.mean(dn_tmp, axis=0))

    df = np.max(dfs, axis=0)
    dn = np.mean(dns, axis=0)

    # Calculate the vibration intensity
    return df / dn


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

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)

    # dist_plot = DistancePlot(0)
    # dist_plot.resize(600, 600)
    # dist_plot.show()

    vib_plot = VibrationIntensityHeatmap(
        start_freq=MIN_FREQ, range_res=config["range_res"]
    )
    vib_plot.resize(600, 600)
    vib_plot.show()

    # Read the saved data file
    data = np.load(args.data)["data"]

    for frame in data:

        vibration_intensity_data = []

        # Average all the receivers
        frame = np.mean(frame, axis=2)

        # Get the fft of the data (n_chirps_per_frame, n_bins, n_receivers)
        fft_result = fft(frame, axis=0)

        # Only take the first half of the fft result
        fft_result = fft_result[:, : SAMPLES_PER_CHIRP // 2]

        def process_frequency(i):
            # Remove random noise
            # processed = random_noise_elimination(
            #     fft_result, i, config["chirp_sampling_rate"]
            # )

            # Calculate the vibration intensity
            vib = vibration_intensity(fft_result, i, config["chirp_sampling_rate"])

            return vib

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            vibration_intensity_data = list(
                executor.map(
                    lambda i: process_frequency(i),
                    range(MIN_FREQ, MAX_FREQ + 1),
                )
            )

        vib_data = np.array(vibration_intensity_data)

        vib_plot.update(vib_data)

        print("Max Vibration Intensity:", np.max(vib_data))

        app.processEvents()

        # frame = background_subtraction(frame)

        # signal = np.mean(frame, axis=0)

        # fft_result = fft(signal, axis=0)
        # fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        # fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # # Plot the data
        # dist_plot.update(
        #     fft_meters[: SAMPLES_PER_CHIRP // 2],
        #     np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        # )

        # app.processEvents()

        # time.sleep(0.1)


if __name__ == "__main__":
    main()
