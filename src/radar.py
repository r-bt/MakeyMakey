from datatime import datetime
from src.xwr.dcapub import DCAPub

class Radar():

    def __init__(self, cfg_path: str):
        """
        Initializes the radar object, starts recording, and publishes the data.

        Args:
            cfg_path (str): Path to the .lua file used in mmWaveStudio to configure the radar
        """

        self.radar = DCAPub(
            cfg=cfg_path,
        )

        try:
            while True:
                frame_data, new_frame = radar.update_frame_buffer()
                
                if new_frame:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    print(f"Frame received at {timestamp}")

        except KeyboardInterrupt:
            print("Stopping radar...")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to the .lua file used in mmWaveStudio"
    )

    args = parser.parse_args()

    radar = Radar(args.cfg)