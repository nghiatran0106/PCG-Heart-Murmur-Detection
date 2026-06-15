import numpy as np
import librosa


def extract_mfcc_features(x, sr: int, n_mfcc: int = 20):
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feats = []

    for arr in [mfcc, delta, delta2]:
        feats.append(arr.mean(axis=1))
        feats.append(arr.std(axis=1))

    return np.concatenate(feats).astype(np.float32)
