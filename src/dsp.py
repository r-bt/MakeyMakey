import numpy as np


def subtract_background(frame):
    after_subtraction = np.zeros_like(frame)
    for i in range(1, frame.shape[0]):
        after_subtraction[i - 1] = frame[i] - frame[i - 1]

    return after_subtraction


def identify_vibrations(heatmap, fft_meters, othermills, max_distance=0.1):
    """
    Takes a heatmap, groups data into clusters and returns the strongest vibrations and their distances

    Args:
        heatmap (np.ndarray): The heatmap to process (vibration_freq_bins, range_bins)
        fft_meters (np.ndarray): The range bins in meters
        othermills (list): A list of distances to other mills
        threshold (int): The threshold to use for identifying vibrations

    Returns:
        The frequencies around each othermill
    """
    objects = []

    for othermill in othermills:
        left_index = np.abs(fft_meters - (othermill - max_distance)).argmin()
        right_index = np.abs(fft_meters - (othermill + max_distance)).argmin()

        # Get a slice of the heatmap around the othermill
        othermill_slice = heatmap[:, left_index : right_index + 1]

        othermill_slice_flat = othermill_slice.flatten()
        othermill_slice_flat = np.hstack([othermill_slice_flat, othermill])

        objects.append(othermill_slice_flat)

    return objects
