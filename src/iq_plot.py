import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np


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
        self.scatter.clear()
        self.scatter.setData(pos=pos)


class VibrationIntensityPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration Intensity Plot")
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.plot_item = self.plot_widget.plot(pen=pg.mkPen(color=(255, 0, 0), width=2))
        self.plot_widget.setLabel("bottom", "Time (s)")
        self.plot_widget.setLabel("left", "Intensity")

    def update(self, data: np.ndarray):
        """Expects data to be a 1D numpy array."""
        if data.size == 0:
            return
        self.plot_item.setData(data)
