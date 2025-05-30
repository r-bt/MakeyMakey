"""
Records data from the DCA1000
"""

import argparse
from datetime import datetime
import csv
import json

from src.radar import Radar

writer = None


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
        writer = csv.DictWriter(f, fieldnames=["data", "timestamp"])
        writer.writeheader()

data = []

def log(msg):
    """
    Callback function to log the data to the csv file
    """
    global data

    data.append(msg)

    # writer.writerow(
    #     {
    #         "data": json.dumps(msg.get("data").tolist()),
    #         "timestamp": msg.get("timestamp"),
    #     }
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initialize the CSV writer
    init_writer()

    # Initialize the radar
    radar = Radar(args.cfg, reshape=False)
    radar.run_polling(cb=log)

    for msg in data:
        writer.writerow(
            {
                "data": json.dumps(msg.get("data").tolist()),
                "timestamp": msg.get("timestamp"),
            }
        )
    
    print("Wrote all data out!")
