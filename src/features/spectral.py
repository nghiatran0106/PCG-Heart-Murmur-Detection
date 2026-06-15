import numpy as np
import librosa


def extract_spectral_features(x, sr: int):
    centroid = librosa.feature.spectral_centroid(y=x, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=x)
    zcr = librosa.feature.zero_crossing_rate(y=x)
    rms = librosa.feature.rms(y=x)

    arrays = [centroid, bandwidth, rolloff, flatness, zcr, rms]

    feats = []

    for arr in arrays:
        feats.extend([
            float(arr.mean()),
            float(arr.std()),
            float(arr.min()),
            float(arr.max()),
        ])

    return np.array(feats, dtype=np.float32)
