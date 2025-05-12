import argparse
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from src.xwr.radar_config import RadarConfig
import queue
from src.dsp import subtract_background
from multiprocessing import Process, Manager
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go


def init_heatmap_dash(CHIRP_RATE, processed_frames, fft_meters):
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("Vibration Heatmap"),
            dcc.Graph(id="heatmap"),
            dcc.Interval(id="interval", interval=1000, n_intervals=0),
        ]
    )

    last_figure = go.Figure(
        data=go.Heatmap(z=np.zeros((100, 100)), colorscale="Hot"),
        layout=go.Layout(
            xaxis=dict(title="Range Bins"), yaxis=dict(title="Vibration Frequency Bins")
        ),
    )

    @app.callback(Output("heatmap", "figure"), Input("interval", "n_intervals"))
    def update_heatmap(_):
        nonlocal last_figure
        if processed_frames.qsize() < 100:
            return last_figure

        processed_frames_list = []
        while len(processed_frames_list) < 100:
            try:
                processed_frames_list.append(processed_frames.get_nowait())
            except queue.Empty:
                break

        p_frames = np.array(processed_frames_list)

        heatmap = []
        for range_bin in range(p_frames.shape[1]):
            time_series = p_frames[:, range_bin]
            f, t, stft_matrix = stft(
                time_series, fs=CHIRP_RATE, nperseg=100, noverlap=16
            )
            magnitude = np.abs(stft_matrix).mean(axis=1)
            heatmap.append(magnitude)

        heatmap = np.array(heatmap).T

        # Threshold
        threshold = 1000
        heatmap = np.where(heatmap > threshold, heatmap, 0)

        step = 10
        tick_vals = np.arange(0, len(fft_meters), step)
        tick_text = [f"{x:.2f}" for x in fft_meters[::step]]

        last_figure = go.Figure(
            data=go.Heatmap(z=heatmap, colorscale="Hot"),
            layout=go.Layout(
                xaxis=dict(
                    title="Range Bins",
                    tickmode="array",
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    tickangle=45,
                ),
                yaxis=dict(title="Vibration Frequency Bins"),
            ),
        )

        return last_figure

    app.run(debug=False)


def main():
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--data", type=str, required=True, help="Path to the .csv file")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the radar
    params = RadarConfig(args.cfg).get_params()

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
        p = Process(
            target=init_heatmap_dash, args=(CHIRP_RATE, processed_frames, fft_meters)
        )
        p.start()

        def process_frame(frame):
            # First get the frame

            # Average across the receivers
            frame = np.mean(frame, axis=2)

            # Apply a hanning window
            window = np.hanning(SAMPLES_PER_CHIRP)
            frame *= window[None, :]  # apply along samples axis

            # Average across the chirps
            frame = np.mean(frame, axis=0)

            # Apply background subtraction
            frame = subtract_background(frame)

            fft_result = fft(frame, axis=0)

            # Only care about the first half of the fft result
            fft_result = fft_result[: SAMPLES_PER_CHIRP // 2]

            processed_frames.put(fft_result)

        # Read the saved data file

        data = np.load(args.data)["data"]

        for frame in data:
            process_frame(frame)

        p.join()


if __name__ == "__main__":
    main()
