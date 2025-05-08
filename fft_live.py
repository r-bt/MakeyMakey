# new imports
import dash
from dash import dcc, html
import plotly.graph_objs as go
import threading
import argparse
from src.radar import Radar
import numpy as np
from src.xwr.radar_config import RadarConfig

fft_data = {"x": [], "y": []}
config = None


def update_frame(msg):
    frame = msg.get("data", None)
    if frame is None or config is None:
        return

    range_res = config["range_res"]
    avg_chirps = np.mean(frame, axis=0)
    signal = avg_chirps[:, 0]

    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    dists = np.arange(fft_magnitude.shape[0]) * range_res

    fft_data["x"] = dists.tolist()
    fft_data["y"] = fft_magnitude.tolist()


def run_dash():
    app = dash.Dash(__name__)
    app.layout = html.Div(
        [
            dcc.Graph(id="live-fft"),
            dcc.Interval(id="interval", interval=500, n_intervals=0),
        ]
    )

    @app.callback(
        dash.Output("live-fft", "figure"), [dash.Input("interval", "n_intervals")]
    )
    def update_fft(n):
        return go.Figure(
            data=[go.Scatter(x=fft_data["x"], y=fft_data["y"], mode="lines")],
            layout=go.Layout(
                title="Live FFT Magnitude",
                xaxis_title="Distance (m)",
                yaxis_title="Magnitude",
            ),
        )

    app.run(debug=False, use_reloader=False)


def main():
    global config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Start the dash server in a thread so it doesn't block
    threading.Thread(target=run_dash, daemon=True).start()

    # Initalize the radar config
    config = RadarConfig(args.cfg).get_params()
    radar = Radar(args.cfg, cb=update_frame)

    # Keep the radar app running
    while True:
        pass


if __name__ == "__main__":
    main()
