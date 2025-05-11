"""
Records data from the DCA1000
"""

import argparse
from datetime import datetime
import csv
import json
import queue
from multiprocessing import Process

from src.radar import Radar

q = queue.Queue()


def write_loop():
    """
    Loop to write the data to the csv file
    """
    filename = "data/radar_data_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["data", "timestamp", "params"])
        writer.writeheader()

        while True:
            msg = q.get()
            if msg is None:  # Exit signal
                break

            # Write the data to the csv file
            writer.writerow(
                {
                    "data": json.dumps(msg["data"].tolist()),
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

    # Start the write loop in a separate process to hopefully reduce packet drops

    p = Process(target=write_loop)
    p.start()

    # Initialize the radar. Don't reshape the data since can't json serialize complex numbers
    radar = Radar(args.cfg, reshape=False)

    radar.run_polling(cb=log)

    # Signal the write loop to exit and wait for the thread to finish
    q.put(None)
    p.join()
