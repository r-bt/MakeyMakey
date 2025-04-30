import pandas as pd
import json
import numpy as np
import cv2


def normalize_and_color(data, max_val=0, cmap=None):
    if max_val <= 0:
        max_val = np.max(data)

    img = (data / max_val * 255).astype(np.uint8)
    if cmap:
        img = cv2.applyColorMap(img, cmap)
    return img


class FrameBuffer:
    # DoubleBuffer for for vis
    def __init__(self, framesize=(256, 256)):
        self.buff = [
            np.zeros(framesize),
            np.zeros(framesize),
        ]

        self.dirty = True
        self.frontbuff = 0

    def write_frame(self, new_frame):
        backbuff = (self.frontbuff + 1) % 2
        self.buff[backbuff] = new_frame
        self.frontbuff = backbuff
        self.dirty = True

    def get_frame(self):
        d = self.dirty
        self.dirty = False
        return d, self.buff[self.frontbuff]


file_path = "data/radar_data_20250430_132508.csv"

n_receivers = 4
samples_per_chirp = 128
n_chirps_per_frame = 128
n_tdm = 0


def fft_processs(adc_samples):
    fft_range = np.fft.fft(adc_samples, axis=1)
    fft_range_doppler = np.fft.fft(fft_range, axis=0)
    fft_range_azi = np.fft.fft(fft_range_doppler, axis=2)

    # fft_mag = np.log(np.abs(fft_range_azi_cd))
    # return fft_range_azi
    # fft_range_azi_cd = np.sum(fft_range_azi, 0)

    fft_mag = np.fft.fftshift(np.log(np.abs(fft_range_doppler[:, :, 0])), axes=0)
    return fft_mag


def reshape_frame(data, n_chirps_per_frame, samples_per_chirp, n_receivers):
    data = data.reshape(-1, 8)  # Assuming we have 4 antennas

    data = data[:, :4] + 1j * data[:, 4:]

    data = data.reshape(n_chirps_per_frame, samples_per_chirp, n_receivers)

    # deinterleve if theres TDM
    if n_tdm > 1:  # TODO: Pretty sure we're not using TDM
        data_i = [data[i::n_tdm, :, :] for i in range(n_tdm)]
        data = np.concatenate(data_i, axis=-1)

    return data


# Read the CSV file
df = pd.read_csv(file_path, chunksize=1)

cv2.namedWindow("fft_viz", cv2.WINDOW_NORMAL)
cv2.resizeWindow("fft_viz", 800, 600)  # Set window size to 800x600 pixels

ax = 2

for chunk in df:
    data_json = json.loads(chunk["data"].iloc[0])

    data = np.array(data_json, dtype=np.int16)

    # Reshape the data
    data = reshape_frame(data, n_chirps_per_frame, samples_per_chirp, n_receivers)

    # Process the data
    fft_mag = fft_processs(data)

    img = normalize_and_color(fft_mag, 18.0, cv2.COLORMAP_WINTER)
    cv2.imshow("fft_viz", img)

    cv2.waitKey(1)

cv2.destroyAllWindows()
