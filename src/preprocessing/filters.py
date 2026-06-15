import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


def resample_if_needed(x, orig_sr: int, target_sr: int):
    if orig_sr == target_sr:
        return x.astype(np.float32), orig_sr

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    y = resample_poly(x, up, down)

    return y.astype(np.float32), target_sr


def bandpass_filter(x, sr: int, low: float, high: float, order: int = 4):
    nyq = 0.5 * sr

    if high >= nyq:
        high = nyq * 0.95

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype="band",
    )

    y = filtfilt(b, a, x)
    return y.astype(np.float32)


def normalize_audio(x, method: str = "zscore", eps: float = 1e-8):
    x = x.astype(np.float32)

    if method == "zscore":
        return (x - np.mean(x)) / (np.std(x) + eps)

    if method == "peak":
        return x / (np.max(np.abs(x)) + eps)

    return x


def pad_or_crop(x, target_len: int):
    if len(x) > target_len:
        return x[:target_len]

    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")

    return x
