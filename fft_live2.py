import argparse
from src.radar import Radar
import numpy as np
from PyQt6 import QtWidgets
from src.distance_plot import DistancePlot
import sys
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

class NoiseRemover:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.M1 = 5 
        self.M2 = 5 
        self.prev_signals = [] 

    def random_noise_elimination(self, signal):
        S1 = np.zeros_like(signal)
        half_window = self.M1 // 2
        
        for i in range(signal.shape[0]):
            start_idx = max(0, i - half_window)
            end_idx = min(signal.shape[0], i + half_window + 1)
            S1[i] = np.mean(signal[start_idx:end_idx], axis=0)
    
        self.prev_signals.append(S1)
        if len(self.prev_signals) > self.M2:
            self.prev_signals.pop(0)
        
        if len(self.prev_signals) > 1:
            S2 = np.zeros_like(S1)
            for sig in self.prev_signals:
                S2 += sig
            S2 /= len(self.prev_signals)
            return S2
        else:
            return S1
    
    def baseline_drift_elimination(self, signal, period_samples):
        if period_samples <= 1:
            return signal
            
        S3 = np.zeros_like(signal)
        half_period = period_samples // 2
        
        for i in range(signal.shape[0]):
            start_idx = max(0, i - half_period)
            end_idx = min(signal.shape[0], i + half_period + 1)
            period_avg = np.mean(signal[start_idx:end_idx], axis=0)
            S3[i] = signal[i] - period_avg
            
        return S3
    
    def estimate_vibration_period(self, signal):
        signal_mag = np.linalg.norm(signal, axis=1) #if signal.ndim > 1 else np.abs(signal)
        peaks, _ = find_peaks(signal_mag, distance=3)
        
        if len(peaks) >= 2:
            peak_diffs = np.diff(peaks)
            return int(np.mean(peak_diffs))
        else:
            return min(len(signal) // 4, 10)
    
    def process_frame(self, frame):
        signal = frame.astype(complex)
        
        signal_filtered = self.random_noise_elimination(signal)
        period = self.estimate_vibration_period(signal_filtered)
        signal_filtered = self.baseline_drift_elimination(signal_filtered, period)
        
        return signal_filtered

def calculate_vibration_intensity(signal, sample_rate, freq_range=(20, 120)):
    fft_result = fft(signal, axis=0)
    fft_freqs = fftfreq(len(signal), 1/sample_rate)
    
    pos_mask = (fft_freqs > 0)
    freqs = fft_freqs[pos_mask]
    fft_vals = fft_result[pos_mask]
    
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs = freqs[freq_mask]
    fft_vals = fft_vals[freq_mask]
    
    magnitudes = np.abs(fft_vals)
    vibration_intensity = magnitudes / np.mean(magnitudes) if np.mean(magnitudes) > 0 else magnitudes
    
    return freqs, vibration_intensity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Initalize the radar
    radar = Radar(args.cfg)

    params = radar.params

    c = 3e8  # speed of light - m/s
    SAMPLES_PER_CHIRP = params["n_samples"]  # adc number of samples per chirp
    SAMPLE_RATE = params["sample_rate"]  # digout sample rate in Hz
    FREQ_SLOPE = params["chirp_slope"]  # frequency slope in Hz (/s)

    # Initialize the noise remover
    noise_remover = NoiseRemover(SAMPLE_RATE)

    # Initalize the GUI
    app = QtWidgets.QApplication(sys.argv)
    dist_plot = DistancePlot(params["range_res"])
    dist_plot.resize(600, 600)
    dist_plot.show()

    def update_frame(msg):
        frame = msg.get("data", None)
        if frame is None:
            return

        # Get the fft of the data
        frame_filtered = noise_remover.process_frame(frame)
        signal = np.mean(frame_filtered, axis=0)
        
        fft_result = fft(signal, axis=0)
        fft_freqs = fftfreq(SAMPLES_PER_CHIRP, 1 / SAMPLE_RATE)
        fft_meters = fft_freqs * c / (2 * FREQ_SLOPE)

        # Plot the data
        dist_plot.update_plot(
            fft_meters[: SAMPLES_PER_CHIRP // 2],
            np.abs(fft_result[: SAMPLES_PER_CHIRP // 2, :]),
        )

    # Initialize the radar
    radar.run_polling(cb=update_frame)


if __name__ == "__main__":
    main()