import argparse
import numpy as np
from src.xwr.radar_config import RadarConfig
import pandas as pd
import json
from src.dsp import reshape_frame


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
        data_json = json.loads(chunk["data"].iloc[0])

        data = np.array(data_json, dtype=np.int16)

        reshaped_frame = reshape_frame(
            data,
            n_chirps_per_frame,
            samples_per_chirp,
            n_receivers,
        )

        # Apply an FFT to the frame across the samples
        fft_data = np.fft.fft(reshaped_frame, axis=1)

