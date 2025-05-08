from src.xwr.radar_config import RadarConfig
import argparse
import numpy as np
import pandas as pd
import json
from src.dsp import reshape_frame
from sklearn import svm


def split(data, n_train):
    """
    Split the data into training and testing sets.

    Args:
        data (np.ndarray): The input data array.
        n_train (int): The percentage of data to use for training.

    Returns:
        tuple: Training and testing data.
    """
    n_train_on = int(len(data) * n_train)
    n_test = len(data) - n_train_on

    train_data = data[:n_train_on]
    test_data = data[n_train_on:]

    return train_data, test_data


if __name__ == "__main__":
    cfg = "scripts/1443_mmwavestudio_config.lua"
    on_data = "data/always_on.csv"
    off_data = "data/always_off.csv"

    # Initalize the radar config
    config = RadarConfig(cfg).get_params()

    n_chirps_per_frame = config["n_chirps"]
    n_receivers = config["n_rx"]
    samples_per_chirp = config["n_samples"]

    # Read in the on_frames
    on_df = pd.read_csv(on_data)
    off_df = pd.read_csv(off_data)

    on_frames = np.array(
        [
            np.fft.fft(
                reshape_frame(
                    np.array(json.loads(row), dtype=np.int16),
                    n_chirps_per_frame,
                    samples_per_chirp,
                    n_receivers,
                ),
                axis=1,
            )
            for row in on_df["data"]
        ]
    )

    off_frames = np.array(
        [
            np.fft.fft(
                reshape_frame(
                    np.array(json.loads(row), dtype=np.int16),
                    n_chirps_per_frame,
                    samples_per_chirp,
                    n_receivers,
                ),
                axis=1,
            )
            for row in on_df["data"]
        ]
    )

    # Now split into training and testing data (80% training, 20% testing)

    on_train, on_test = split(on_frames, 0.8)
    off_train, off_test = split(off_frames, 0.8)

    # Now we need to create the labels
    on_labels = np.ones(len(on_train))
    off_labels = np.zeros(len(off_train))

    # Now we need to combine the data and labels
    train_data = np.concatenate((on_train, off_train), axis=0)
    train_labels = np.concatenate((on_labels, off_labels), axis=0)

    # Split up the real and imaginary parts
    train_data = np.concatenate(
        (
            np.real(train_data),
            np.imag(train_data),
        ),
        axis=1,
    )

    # Now train an SVM model
    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(train_data, train_labels)
    print("Model trained")

    # Now we need to test the model
    on_test_labels = np.ones(len(on_test))
    off_test_labels = np.zeros(len(off_test))

    # Now we need to combine the data and labels
    test_data = np.concatenate((on_test, off_test), axis=0)

    test_data = np.concatenate(
        (
            np.real(test_data),
            np.imag(test_data),
        ),
        axis=1,
    )

    test_labels = np.concatenate((on_test_labels, off_test_labels), axis=0)
    print("Testing model")

    # Now we need to test the model
    predictions = clf.predict(test_data)
    print("Model tested")

    # Now we need to calculate the accuracy
    accuracy = np.sum(predictions == test_labels) / len(test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Model accuracy calculated")
