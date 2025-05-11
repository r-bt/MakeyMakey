from datetime import datetime
from src.xwr.dcapub import DCAPub
from src.xwr.dsp import reshape_frame
import argparse


class Radar:

    def __init__(self, cfg_path: str, reshape=True):
        """
        Initializes the radar object, starts recording, and publishes the data.

        Args:
            cfg_path (str): Path to the .lua file used in mmWaveStudio to configure the radar
            reshape (bool): Whether to reshape the data or not. Default is True.
        """
        print(f"[INFO] Starting radar node with config: {cfg_path}")

        self.radar = DCAPub(
            cfg=cfg_path,
        )

        self.config = self.radar.config
        self.params = self.radar.params
        print("[INFO] Radar connected. Params:")
        print(self.radar.config)

        self.reshape = reshape

    def run_polling(self, cb=None):
        print("[INFO] Begin capturing data!")

        # Flush the data socket to clear any old data
        self.radar.dca1000.flush_data_socket()

        try:
            while True:
                frame_data, new_frame = self.radar.update_frame_buffer()

                if new_frame:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # If reshaping is enabled, reshape the frame data
                    if self.reshape:
                        frame_data = reshape_frame(
                            frame_data,
                            self.params["n_chirps"],
                            self.params["n_samples"],
                            self.params["n_rx"],
                        )

                    msg = {
                        "data": frame_data,
                        "timestamp": timestamp,
                        "params": self.params,
                    }

                    if cb:
                        cb(msg)

        except KeyboardInterrupt:
            self.close()
            print("[INFO] Stopping radar...")

    def read(self):
        """
        Reads single frame of data from the radar.
        """

        # Flush the data socket to clear any old data
        self.radar.dca1000.flush_data_socket()

        second = False

        try:
            while True:
                frame_data, new_frame = self.radar.update_frame_buffer()

                if new_frame:
                    if not second:
                        second = True
                    else:
                        if self.reshape:
                            frame_data = reshape_frame(
                                frame_data,
                                self.params["n_chirps"],
                                self.params["n_samples"],
                                self.params["n_rx"],
                            )

                        return frame_data
        except KeyboardInterrupt:
            print("[INFO] Stopping frame capture...")

    def flush(self):
        """
        Flushes the data socket to clear any old data.
        """
        self.radar.dca1000.flush_data_socket()

    def close(self):
        """
        Closes the radar connection and stops capturing data.
        """
        self.radar.dca1000.close()
        print("[INFO] Radar connection closed.")


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
