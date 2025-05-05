See 24.6 in the mmwave studio user guide for details about how the data from the DCA1000 is stored

For xWR12xx/xWR14xx, only interleaved mode is supported

## Data Format

Every `PERIODICITY` we transmit `CHRIP_LOOPS` chirps. Each chirp linearly sweeps from `START_FREQ` onwards. We capture 
`ADC_SAMPLES` samples during this chrip across four different RX antennas.

