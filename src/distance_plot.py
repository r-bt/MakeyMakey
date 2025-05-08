import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import sys


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


if __name__ == "__main__":
    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    iq_plot = DistancePlot(range_res=0.1)  # Example range resolution
    iq_plot.resize(600, 600)
    iq_plot.show()

    # Example data
    data = np.linspace(0, 10, 100)  # 1D array of distance values

    iq_plot.update(data)

    # Start the Qt event loop
    sys.exit(app.exec_())
