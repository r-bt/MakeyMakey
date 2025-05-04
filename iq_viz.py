import argparse
from src.radar import Radar
import numpy as np
from src.dsp import reshape_frame
import matplotlib.pyplot as plt

n_receivers = 4
samples_per_chirp = 128
n_chirps_per_frame = 128


def update_frame(msg):
    reshaped_frame = reshape_frame(
        np.array(msg['data'], dtype=np.int16), n_chirps_per_frame, samples_per_chirp, n_receivers
    )

    # Plot the IQ samples
    iq = reshaped_frame.reshape(-1)  # shape: (n_chirps * n_samples * n_rx,)
    plt.figure(figsize=(6, 6))
    plt.plot(iq.real, iq.imag, ".", markersize=1)
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title("IQ Samples on XY Plane")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initialize the radar
    radar = Radar(args.cfg, cb=update_frame)
