# Seismic denoiser
![A buried wave, before and after](figures/cover.png)

## Overview

Seismology is built on signal processing, which ties it to many of my other projects, and Earth systems are a big passion of mine. Southern California has about 10,000 earthquakes a year, most of them too small to feel, and under Los Angeles many are buried in traffic and other noise, so California has a wealth of seismic data to work with.

The aim was to compare denoising methods from geophysics and seismology with some that are more common in other disciplines, and to find which preserved the seismologically relevant features of a recording, the P-wave arrival time and the first motion (whether the ground moves up or down first). Bandpass filters, wavelets and a random-matrix denoiser were scored on 3,000 of the noisiest labelled recordings from the Southern California Seismic Network. I also stacked recordings from a simulated array of sensors with minimum-variance weights, to denoise further and make the signal more coherent.

Each method was scored with two standard seismological tools, run on the denoised recordings and on the raw recordings for comparison: an STA/LTA picker, which flags the arrival where the energy in a short window jumps relative to a longer one, and a first-motion check. Every recording in the dataset has its arrival at the same point, so the arrival times were first moved at random. The picker then found 56% of arrivals on the raw recordings and 60% after the random-matrix denoiser, barely an improvement, as the denoiser keeps what the recordings have in common, which only helped while the arrivals were lined up. A plain bandpass filter (keeping 1-20 Hz, run forwards and backwards so the arrival is not shifted) raised this to 63% without changing any first motions. Wavelet thresholding picked the most arrivals (68%) and added about 3 dB of signal-to-noise, but flipped the first motion on about one trace in fourteen, so a cleaner-looking waveform can be worse for picking.

The array study was quite interesting. Minimum-variance weights, which give the least noise for the way the noise is correlated between sensors, are really useful for cancelling noise shared between neighbouring sensors, reaching 13.9 dB of noise reduction against 10.9 dB for a plain average of 32 sensors. The weights need the noise covariance between the sensors, estimated from a stretch of recorded noise. Estimated raw, it takes about 0.8 s of noise before the weights beat a plain average, but cleaned with shrinkage from my correlation library (corrlib), 0.2 s is enough.

| | |
|:-:|:-:|
| ![One trace](figures/fig1_one_trace.png) | ![Scoreboard](figures/fig2_scoreboard.png) |
| ![The array](figures/fig3_array.png) | ![Crossover](figures/fig4_crossover.png) |
| ![Gain against array size](figures/hero.png) | |

**Built:** an STA/LTA picker and a first-motion estimator, both calibrated on quiet traces (99% and 97%); a test with randomly moved arrival times; a synthetic array with shared noise; held-out scoring.

`seismic.py` · `seismic_denoiser.ipynb` · `make_cover.py` · `make_pick_errors.py` · Python, NumPy, SciPy, h5py, PyWavelets · uses [`corrlib`](../004-corrlib-correlation-toolbox)

*A side project, built for fun. Labelled waveforms from Ross, Meier & Hauksson (2018), [doi:10.1029/2017JB015251](https://doi.org/10.1029/2017JB015251); Southern California Seismic Network data via the SCEDC, [doi:10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1). The data isn't included in the repository (6 GB).*
