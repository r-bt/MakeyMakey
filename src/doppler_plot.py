import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import numpy as np
import sys


class DopplerPlot(QtWidgets.QMainWindow):
    """
    A line plot of multiple distance vs. intensity curves.
    """

    def __init__(self, range_res: float, num_lines: int = 4):
        super().__init__()

        self.range_res = range_res
        self.setWindowTitle("Doppler Plot")
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        colors = [(255, 0, 0), (0, 255, 0), (0, 150, 255), (255, 255, 0)]
        self.lines = [
            pg.PlotDataItem(pen=pg.mkPen(color=colors[i], width=2), symbol=None)
            for i in range(num_lines)
        ]

        for line in self.lines:
            self.plot_widget.addItem(line)

        self.plot_widget.setLabel("bottom", "Frequency (Hz)")
        self.plot_widget.setLabel("left", "Magnitude")

    def update(self, distances: np.ndarray, data: np.ndarray):
        """
        Args:
            data_list (np.ndarray): 2D array of (n_distances, n_receivers).
        """
        if data is None:
            return

        for i in range(data.shape[1]):
            self.lines[i].setData(distances, data[:, i])

        self.plot_widget.setXRange(0, 3.5)
        global_max = np.max(np.max(data, axis=0), axis=0)
        self.plot_widget.setYRange(0, 5000)


if __name__ == "__main__":
    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    iq_plot = DistancePlot(range_res=0.1)  # Example range resolution
    iq_plot.resize(600, 600)
    iq_plot.show()

    distances = np.linspace(0, 5, 90)

    # Add 10 entries between 5 and 10
    distances = np.concatenate((distances, np.linspace(5, 10, 10)))

    # Example data
    base = np.linspace(0, 10, 100)
    data_list = [base, base * 0.8, np.sin(base), np.exp(-0.1 * base) * 10]

    data = np.column_stack(data_list)

    iq_plot.update(distances, data)

    # sys.exit(app.exec_())
    sys.exit(app.exec())
