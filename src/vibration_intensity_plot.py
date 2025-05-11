import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
import numpy as np


class VibrationIntensityHeatmap(QtWidgets.QMainWindow):
    def __init__(self, start_freq: float, range_res: float):
        super().__init__()

        self.start_freq = start_freq
        self.range_res = range_res

        self.setWindowTitle("Vibration Intensity Heatmap")

        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.plot_widget.setLabel("bottom", "Frequency (Hz)")
        self.plot_widget.setLabel("left", "Distance (m)")
        self.plot_widget.setAspectLocked(False)

        # Optional: color map
        cbi = pg.ColorBarItem(colorMap="CET-L3")
        plot_item = self.plot_widget.getPlotItem()
        cbi.setImageItem(self.img_item, insert_in=plot_item)

    def update(self, data: np.ndarray):
        if data.ndim != 2:
            raise ValueError("Expected 2D array (n_bins, n_freqs)")

        n_freqs, n_freq_bins = data.shape

        self.plot_widget.setXRange(self.start_freq, self.start_freq + n_freqs)
        self.plot_widget.setYRange(0, n_freq_bins * self.range_res)

        self.img_item.setImage(data, levels=(0, 2000))

        image_rect = QtCore.QRectF(
            self.start_freq,
            0,
            n_freqs,
            n_freq_bins * self.range_res,
        )
        self.img_item.setRect(image_rect)
