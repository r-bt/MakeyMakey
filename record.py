import argparse
import queue
import threading
import numpy as np
from datetime import datetime
import os

from src.radar import Radar

log_queue = queue.Queue()
stop_event = threading.Event()

buffer = []


def writer_thread(filename):
    while not stop_event.is_set() or not log_queue.empty():
        try:
            item = log_queue.get(timeout=0.05)
            buffer.append((item["timestamp"], item["data"]))
        except queue.Empty:
            continue

    flush(buffer, filename)


def flush(buffer, filename):
    timestamps, data = zip(*buffer)
    np.savez_compressed(filename, data=np.stack(data), timestamp=np.array(timestamps))


def log(msg):
    """
    Callback function to log the data to the csv file
    """

    log_queue.put(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the npz writer
    filename = f"data/radar_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    os.makedirs("data", exist_ok=True)

    thread = threading.Thread(target=writer_thread, args=(filename,))
    thread.start()

    # Initialize the radar
    radar = Radar(args.cfg)
    radar.run_polling(cb=log)

    stop_event.set()
    thread.join()
