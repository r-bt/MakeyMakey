"""
Records data from the DCA1000
"""

import argparse
from datetime import datetime
import csv
import json
import threading
import queue
import numpy as np

from src.radar import Radar

writer = None

q = queue.Queue()


def init_writer():
    """
    Initializes the csv writer
    """
    global writer
    if writer is None:
        filename = "data/radar_data_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        f = open(filename, "w", newline="")
        writer = csv.DictWriter(
            f, fieldnames=["data_real", "data_imag", "timestamp", "params"]
        )
        writer.writeheader()


def write_loop():
    """
    Loop to write the data to the csv file
    """
    global writer
    while True:
        msg = q.get()
        if msg is None:
            break

        # Write the data to the csv file
        writer.writerow(
            {
                "data_real": json.dumps(np.real(msg["data"]).tolist()),
                "data_imag": json.dumps(np.imag(msg["data"]).tolist()),
                "timestamp": msg["timestamp"],
                "params": json.dumps(msg["params"]),
            }
        )


def log(msg):
    """
    Callback function to log the data to the csv file
    """
    q.put(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initialize the CSV writer
    init_writer()

    # Start the write loop in a separate thread
    write_thread = threading.Thread(target=write_loop)
    write_thread.start()

    # Initialize the radar
    radar = Radar(args.cfg)

    radar.run_polling(cb=log)
