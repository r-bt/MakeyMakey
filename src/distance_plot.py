import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import sys


class DistancePlot(QtWidgets.QMainWindow):
    """
    A line plot of multiple distance vs. intensity curves.
    """

    def __init__(self, range_res: float, num_lines: int = 4):
        super().__init__()

        self.range_res = range_res
        self.setWindowTitle("Distance Plot")
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        colors = [(255, 0, 0), (0, 255, 0), (0, 150, 255), (255, 255, 0)]
        self.lines = [
            pg.PlotDataItem(pen=pg.mkPen(color=colors[i], width=2), symbol=None)
            for i in range(num_lines)
        ]

        for line in self.lines:
            self.plot_widget.addItem(line)

        self.plot_widget.setLabel("bottom", "Distance (m)")
        self.plot_widget.setLabel("left", "Intensity")

    def update(self, distances: np.ndarray, data_list: list[np.ndarray]):
        """
        Args:
            data_list (list[np.ndarray]): List of 1D arrays of intensity values.
        """
        if not data_list:
            return

        for i, data in enumerate(data_list):
            if i >= len(self.lines):
                break
            self.lines[i].setData(distances[: len(data)], data)

        self.plot_widget.setXRange(0, distances[-1] * 1.1)
        global_max = max((np.max(data) for data in data_list if data.size), default=1)
        self.plot_widget.setYRange(0, global_max * 1.1)


if __name__ == "__main__":
    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    iq_plot = DistancePlot(range_res=0.1)  # Example range resolution
    iq_plot.resize(600, 600)
    iq_plot.show()

    distances = np.linspace(0, 5, 90)

    # Add 10 entries between 5 and 10
    distances = np.concatenate((distances, np.linspace(5, 10, 50)))

    # Example data
    base = np.linspace(0, 10, 100)
    data_list = [base, base * 0.8, np.sin(base), np.exp(-0.1 * base) * 10]

    iq_plot.update(data_list, distances)

    sys.exit(app.exec_())
