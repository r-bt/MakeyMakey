import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np


class DistancePlot(QtWidgets.QMainWindow):
    """
    A line plot of the distance vs. intensity.
    """

    def __init__(self, range_res: float):
        """
        Args:
            range_res (float): The range resolution of the radar.
        """
        super().__init__()

        self.range_res = range_res

        self.setWindowTitle("Distance Plot")
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.line = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 255, 255), width=2), symbol=None
        )
        self.plot_widget.addItem(self.line)
        self.plot_widget.setLabel("bottom", "Distance (m)")
        self.plot_widget.setLabel("left", "Intensity")

    def update(self, data: np.ndarray):
        """
        Args:
            data (np.ndarray): 1D array of distance values.
        """
        if data.size == 0:
            return
        self.line.setData(data)
        self.plot_widget.setXRange(0, len(data) * self.range_res)
        self.plot_widget.setYRange(0, np.max(data) * 1.1)


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

        self.line = pg.PlotDataItem(
            pen=pg.mkPen(color=(0, 255, 255), width=2), symbol=None
        )
        self.plot_widget.addItem(self.line)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel("bottom", "Frequency (Hz)")
        self.plot_widget.setLabel("left", "Intensity")

    def update(self, points: list[tuple[float, float]]):
        if not points:
            return
        self.line.setData(*zip(*points))



