import argparse
from src.radar import Radar
import numpy as np
from src.dsp import reshape_frame
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import sys
import threading
import queue

q = queue.Queue()


class IQPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live IQ Plot")
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.scatter = pg.ScatterPlotItem(
            size=2, pen=None, brush=pg.mkBrush(0, 255, 255, 150)
        )
        self.plot_widget.addItem(self.scatter)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel("bottom", "In-phase (I)")
        self.plot_widget.setLabel("left", "Quadrature (Q)")

    def update(self, iq_data: np.ndarray):
        """Expects iq_data to be a 1D complex numpy array."""
        if iq_data.size == 0:
            return
        pos = np.column_stack((iq_data.real, iq_data.imag))
        self.scatter.setData(pos=pos)


n_receivers = 4
samples_per_chirp = 128
n_chirps_per_frame = 128


def update_frame(msg):
    reshaped_frame = reshape_frame(
        np.array(msg["data"], dtype=np.int16),
        n_chirps_per_frame,
        samples_per_chirp,
        n_receivers,
    )

    global iq_plot
    # reshape if needed
    iq = reshaped_frame.reshape(-1)  # Flatten the array

    q.put(iq)  # Put the data in the queue


def radar_thread(cfg, callback):
    """
    This function runs in a separate thread to handle the radar data.
    It initializes the radar and starts receiving data.
    """
    radar = Radar(cfg, cb=callback)


# Global IQPlot instance for access in callback
iq_plot = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    iq_plot = IQPlot()
    iq_plot.resize(600, 600)
    iq_plot.show()

    # Initialize the radar
    radar_thread_instance = threading.Thread(
        target=radar_thread, args=(args.cfg, update_frame), daemon=True
    )
    radar_thread_instance.start()

    # Main loop to update the plot
    while True:
        if not q.empty():
            iq_data = q.get()
            iq_plot.update(iq_data)

        # Process Qt events
        app.processEvents()

        # Sleep for a short duration to avoid busy waiting
        QtCore.QThread.msleep(10)

    sys.exit(app.exec_())
