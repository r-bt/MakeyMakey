This is a WIP of how I understand what we're doing based on the HomeOSD paper

# Pipeline

The IWR1443 and the DCA1000 provide complex-valued samples of the IF signal which is the result of mixing the RX and TX signal and putting it through a lower-pass filter i.e. it's the beat frequency.

This gives us IQ = [I_1 + jQ_1, I_2 + jQ_2, ...] where I_n and Q_n are the in-phase and quadrature components

Then we want to compute the FFT

S(k,f) = FFT(IQ)

For any frequency bin we can get the real and imaginary components of that frequency bin as

I(k,f) = Re(S(k,f))
Q(k,f) = Im(S(k,f))



