import argparse
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from src.radar import Radar
import queue
from multiprocessing import Process, Manager
from src.dsp import subtract_background, identify_vibrations
import joblib
import cv2

model = joblib.load("svm_model.joblib")

OTHERMILL_DISTANCES = [2.35, 3.9]
DISTANCE_THRESHOLD = 0.1

alpha = 0.6  # decay factor for running average
background = None  # initialize

othermill_available_img = cv2.imread("othermill_available.png")
othermill_unavailable_img = cv2.imread("othermill_unavailable.png")

window_of_predictions = [[0 for _ in range(len(OTHERMILL_DISTANCES))] for _ in range(3)]
window_index = 0


def subtract_background_1(current_frame):
    global background
    if background is None:
        background = current_frame.copy()
        return np.zeros_like(current_frame)
    background = alpha * background + (1 - alpha) * current_frame
    return current_frame - background.astype(current_frame.dtype)


def init_plot(CHIRP_RATE, processed_frames, fft_meters, chunk_size=128):
    while True:
        if processed_frames.qsize() < chunk_size:
            continue

        processed_frames_list = []
        while len(processed_frames_list) < chunk_size:
            try:
                processed_frames_list.append(processed_frames.get_nowait())
            except queue.Empty:
                break

        p_frames = np.array(processed_frames_list)

        heatmap = []
        for range_bin in range(p_frames.shape[1]):
            time_series = p_frames[:, range_bin]
            f, t, stft_matrix = stft(
                time_series, fs=CHIRP_RATE, nperseg=64, noverlap=16
            )
            magnitude = np.abs(stft_matrix).mean(axis=1)
            heatmap.append(magnitude)

        heatmap = np.array(heatmap).T  # shape: (vibration_freq_bins, range_bins)

        # Detect the objects
        objects = identify_vibrations(
            heatmap,
            fft_meters,
            othermills=OTHERMILL_DISTANCES,
            max_distance=DISTANCE_THRESHOLD,
        )

        predictions = model.predict(objects)

        global window_index
        global window_of_predictions

        window_of_predictions[window_index] = predictions
        window_index = (window_index + 1) % len(window_of_predictions)

        avg_predictions = np.mean(window_of_predictions, axis=0)

        print(avg_predictions)

        # Now show an image of whether each Othermill is vibrating or not with opencv
        obj_imgs = []
        for i, obj in enumerate(objects):
            if avg_predictions[i] == 1:
                img = othermill_available_img
            else:
                img = othermill_unavailable_img

            obj_imgs.append(img)

        combined = np.hstack(obj_imgs)
        cv2.imshow("Othermill Status", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar
    radar = Radar(args.cfg)

    params = radar.params

    c = 3e8  # speed of light - m/s
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    CHIRP_RATE = params["chirp_sampling_rate"]

    fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
    fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)
    fft_freqs = fft_freqs[: SAMPLES_PER_CHIRP // 2]
    fft_meters = fft_meters[: SAMPLES_PER_CHIRP // 2]

    # Start a new process for matplotlib

    with Manager() as manager:
        processed_frames = manager.Queue()
        p = Process(target=init_plot, args=(CHIRP_RATE, processed_frames, fft_meters))
        p.start()

        def process_frame(msg):
            frame = msg.get("data", None)

            if frame is None:
                return

            # Average across the receivers
            frame = np.mean(frame, axis=2)

            # Apply a hanning window
            window = np.hanning(SAMPLES_PER_CHIRP)
            frame *= window[None, :]  # apply along samples axis

            # Average across the chirps
            frame = np.mean(frame, axis=0)

            # Apply background subtraction
            frame = subtract_background_1(subtract_background(frame))

            fft_result = fft(frame, axis=0)

            # Only care about the first half of the fft result
            fft_result = fft_result[: SAMPLES_PER_CHIRP // 2]

            processed_frames.put(fft_result)

        # Read the saved data file

        radar.run_polling(cb=process_frame)

        p.join()


if __name__ == "__main__":
    main()
