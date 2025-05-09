from datetime import datetime
from src.xwr.dcapub import DCAPub
from src.xwr.radar_config import RadarConfig
from src.dsp import reshape_frame
import argparse


class Radar:

    def __init__(self, cfg_path: str, cb=None):
        """
        Initializes the radar object, starts recording, and publishes the data.

        Args:
            cfg_path (str): Path to the .lua file used in mmWaveStudio to configure the radar
        """

        self.radar = DCAPub(
            cfg=cfg_path,
        )

        print("DCA1000 conected!")

        self.config = RadarConfig(cfg_path).get_params()

    def run_polling(self, cb=None):
        print("Begin capturing data!")
        self.radar.dca1000.flush_data_socket()

        try:
            while True:
                frame_data, new_frame = self.radar.update_frame_buffer()

                if new_frame:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    frame_data = reshape_frame(
                        frame_data,
                        self.config["n_chirps"],
                        self.config["n_samples"],
                        self.config["n_rx"],
                    )

                    msg = {"data": frame_data, "timestamp": timestamp}

                    if cb:
                        cb(msg)

        except KeyboardInterrupt:
            print("Stopping radar...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Path to the .lua file used in mmWaveStudio",
    )

    args = parser.parse_args()

    radar = Radar(args.cfg)
