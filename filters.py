# filters.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# =========================
# Bandpass (1–40 Hz)
# =========================
def bandpass_filter(data, fs=250, low=1, high=40, order=4):
    """
    Aplica un filtro paso banda Butterworth.
    - fs: frecuencia de muestreo (Hz)
    - low, high: frecuencias de corte (Hz)
    - order: orden del filtro
    """
    nyq = 0.5 * fs
    low_cut = low / nyq
    high_cut = high / nyq
    b, a = butter(order, [low_cut, high_cut], btype="band")
    return filtfilt(b, a, data)

# =========================
# Notch (50 Hz)
# =========================
def notch_filter(data, fs=250, freq=50, quality=30):
    """
    Aplica un notch (filtro elimina banda) en la frecuencia especificada.
    - fs: frecuencia de muestreo (Hz)
    - freq: frecuencia central a eliminar (Hz)
    - quality: factor Q (más alto = notch más estrecho)
    """
    nyq = 0.5 * fs
    w0 = freq / nyq
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, data)

# =========================
# Pipeline completo
# =========================
def preprocess_signal(data, fs=250):
    """
    Preprocesa la señal de un único canal (CH1):
    1. Bandpass 1–40 Hz
    2. Notch 50 Hz
    """
    if len(data) < 10:
        return data  # evita errores con buffers demasiado pequeños

    filtered = bandpass_filter(data, fs=fs, low=1, high=40, order=4)
    filtered = notch_filter(filtered, fs=fs, freq=50, quality=30)
    return filtered
