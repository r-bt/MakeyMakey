import argparse
from src.radar import Radar
import numpy as np
from src.dsp import reshape_frame
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

n_receivers = 4
samples_per_chirp = 128
n_chirps_per_frame = 128

# --- Setup live plot window ---
app = QtGui.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="Live IQ Plot")
plot = win.addPlot(title="IQ Samples on XY Plane")
scatter = pg.ScatterPlotItem(
    size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120)
)
plot.addItem(scatter)
plot.setLabel("left", "Quadrature (Q)")
plot.setLabel("bottom", "In-phase (I)")
plot.setAspectLocked(True)
win.resize(600, 600)


def update_frame(msg):
    reshaped_frame = reshape_frame(
        np.array(msg["data"], dtype=np.int16),
        n_chirps_per_frame,
        samples_per_chirp,
        n_receivers,
    )

    # Plot the IQ samples
    iq = reshaped_frame.reshape(-1)
    scatter.setData(pos=np.column_stack((iq.real, iq.imag)))
    QtGui.QApplication.processEvents()  # flush GUI updates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initialize the radar
    radar = Radar(args.cfg, cb=update_frame)
    QtGui.QApplication.instance().exec_()
