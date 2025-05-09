import argparse
from src.radar import Radar
import numpy as np
from src.xwr.radar_config import RadarConfig
from PyQt5 import QtWidgets
from src.distance_plot import DistancePlot
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(config["range_res"])
    dist_plot.resize(600, 600)
    dist_plot.show()

    def update_frame(msg):
        frame = msg.get("data", None)
        if frame is None:
            return

        avg_chirps = np.mean(frame, axis=0)
        signal = avg_chirps[:, 0]

        samples_per_chirp = signal.shape[0]

        fft_result = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_result[: samples_per_chirp // 2])

        dist_plot.update(fft_magnitude)

        app.processEvents()

    # Initialize the radar
    radar = Radar(args.cfg, cb=update_frame)
    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()
