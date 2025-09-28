import numpy as np
import mne

# Parámetros de filtrado
HPF_HZ   = 0.5   # pasa altos
LPF_HZ   = 70.0  # pasa bajos
NOTCH_HZ = 50.0  # notch
SFREQ    = 250   # Hz

def compute_fft(uV_signal, sfreq=SFREQ):
    """ Calcula FFT de la señal en µV. """
    x = np.asarray(uV_signal, dtype=float)
    x = x - np.mean(x)
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), d=1.0/sfreq)
    mag = (2.0 / np.sum(w)) * np.abs(X)
    if len(mag) > 0:
        mag[0] = mag[0] / 2.0
    return f.tolist(), mag.tolist()

def process_ch1(ch1_uV, sfreq=SFREQ):
    """ Filtra CH1 y calcula su FFT. """
    # Notch 50 Hz
    ch1_nf = mne.filter.notch_filter(
        ch1_uV, Fs=sfreq, freqs=[NOTCH_HZ], verbose='ERROR'
    )
    # Banda 0.5–70 Hz
    ch1_bp = mne.filter.filter_data(
        ch1_nf, sfreq, HPF_HZ, LPF_HZ, verbose='ERROR'
    )
    # FFT
    f, mag = compute_fft(ch1_bp, sfreq)
    return {
        "ch1_filtered": ch1_bp.tolist(),
        "fft": {"f": f, "mag": mag},
    }
