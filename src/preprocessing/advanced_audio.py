from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt, medfilt


def to_mono(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def resample_audio(x, orig_sr, target_sr):
    x = to_mono(x)
    if orig_sr == target_sr:
        return x.astype(np.float32)
    y = librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)
    return y.astype(np.float32)


def zscore_normalize(x):
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-8:
        return x.astype(np.float32)
    return ((x - mean) / std).astype(np.float32)


def bandpass_filter(x, sr, low=25.0, high=800.0, order=4):
    x = np.asarray(x, dtype=np.float32)
    nyq = 0.5 * sr
    low = max(0.001, low / nyq)
    high = min(0.999, high / nyq)

    if low >= high:
        return x.copy()

    b, a = butter(order, [low, high], btype="band")

    # filtfilt cần tín hiệu đủ dài
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(x) <= padlen:
        return x.copy()

    y = filtfilt(b, a, x)
    return y.astype(np.float32)


def spike_reduction(x, kernel_size=5, clip_sigma=4.0):
    """
    Giảm spike bằng median-based clipping.
    """
    x = np.asarray(x, dtype=np.float32)

    if kernel_size % 2 == 0:
        kernel_size += 1

    med = medfilt(x, kernel_size=kernel_size)
    resid = x - med

    mad = np.median(np.abs(resid)) / 0.6745 if np.median(np.abs(resid)) > 0 else 0.0
    thr = clip_sigma * (mad + 1e-8)

    y = x.copy()
    mask = np.abs(resid) > thr
    y[mask] = med[mask]
    return y.astype(np.float32)


def spectral_gating(
    x,
    sr,
    n_fft=1024,
    hop_length=160,
    win_length=400,
    noise_sec=0.5,
    n_std_thresh=1.5,
):
    """
    Simple spectral gating:
    - Ước lượng noise từ đoạn đầu noise_sec giây
    - Tạo ngưỡng theo mean + std
    - Làm suy giảm các thành phần dưới ngưỡng
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) < win_length:
        return x.copy()

    D = librosa.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
    )

    mag = np.abs(D)
    phase = np.exp(1j * np.angle(D))

    noise_frames = max(1, int(noise_sec * sr / hop_length))
    noise_frames = min(noise_frames, mag.shape[1])

    noise_mag = mag[:, :noise_frames]
    noise_mean = noise_mag.mean(axis=1, keepdims=True)
    noise_std = noise_mag.std(axis=1, keepdims=True)

    threshold = noise_mean + n_std_thresh * noise_std

    # mask mềm: phần dưới ngưỡng bị giảm mạnh
    mask = mag >= threshold
    mag_clean = np.where(mask, mag - noise_mean, 0.15 * mag)

    D_clean = mag_clean * phase
    y = librosa.istft(
        D_clean,
        hop_length=hop_length,
        win_length=win_length,
        length=len(x),
    )
    return y.astype(np.float32)


def parse_circor_tsv(tsv_path):
    """
    Parse file .tsv của CirCor.
    Kỳ vọng ít nhất 3 cột: start_sec, end_sec, state
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        return None

    try:
        df = pd.read_csv(tsv_path, sep="\t", header=None)
    except Exception:
        return None

    if df.shape[1] < 2:
        return None

    if df.shape[1] >= 3:
        df = df.iloc[:, :3].copy()
        df.columns = ["start_sec", "end_sec", "state"]
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ["start_sec", "end_sec"]
        df["state"] = -1

    df["start_sec"] = pd.to_numeric(df["start_sec"], errors="coerce")
    df["end_sec"] = pd.to_numeric(df["end_sec"], errors="coerce")
    df = df.dropna(subset=["start_sec", "end_sec"]).reset_index(drop=True)

    return df


def load_circor_cycles(tsv_path, sr):
    """
    Tạo full heart cycles theo logic:
    mỗi cycle = từ một S1 đến ngay trước S1 tiếp theo.
    Nếu không tìm thấy state S1, trả về rỗng.
    """
    df = parse_circor_tsv(tsv_path)
    if df is None or len(df) == 0:
        return []

    state_str = df["state"].astype(str).str.strip().str.lower()
    s1_idx = [i for i, s in enumerate(state_str) if s in {"1", "s1"}]

    cycles = []
    if len(s1_idx) >= 2:
        for i in range(len(s1_idx) - 1):
            s_idx = s1_idx[i]
            e_idx = s1_idx[i + 1]

            start_sec = float(df.loc[s_idx, "start_sec"])
            end_sec = float(df.loc[e_idx, "start_sec"])

            if end_sec > start_sec:
                cycles.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "start_sample": int(round(start_sec * sr)),
                    "end_sample": int(round(end_sec * sr)),
                })

    return cycles


def extract_cycles(x, cycles, min_duration_sec=0.25, max_duration_sec=2.5, sr=4000):
    out = []
    for c in cycles:
        s = max(0, c["start_sample"])
        e = min(len(x), c["end_sample"])
        if e <= s:
            continue
        dur = (e - s) / sr
        if dur < min_duration_sec or dur > max_duration_sec:
            continue
        out.append({
            "signal": x[s:e].astype(np.float32),
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "duration_sec": dur,
        })
    return out


def preprocess_signal(x, orig_sr, target_sr, cfg):
    y = resample_audio(x, orig_sr=orig_sr, target_sr=target_sr)

    y = bandpass_filter(
        y,
        sr=target_sr,
        low=cfg.get("bandpass_low", 25),
        high=cfg.get("bandpass_high", 800),
        order=cfg.get("filter_order", 4),
    )

    if cfg.get("spike_reduction", False):
        y = spike_reduction(
            y,
            kernel_size=cfg.get("spike_kernel_size", 5),
            clip_sigma=cfg.get("spike_clip_sigma", 4.0),
        )

    if cfg.get("spectral_gating", False):
        y = spectral_gating(
            y,
            sr=target_sr,
            n_fft=cfg.get("n_fft", 1024),
            hop_length=cfg.get("hop_length", 160),
            win_length=cfg.get("win_length", 400),
            noise_sec=cfg.get("noise_sec", 0.5),
            n_std_thresh=cfg.get("n_std_thresh", 1.5),
        )

    if cfg.get("normalize", "zscore") == "zscore":
        y = zscore_normalize(y)

    stats = {
        "orig_rms": float(np.sqrt(np.mean(np.square(x)) + 1e-12)),
        "proc_rms": float(np.sqrt(np.mean(np.square(y)) + 1e-12)),
        "orig_len": int(len(x)),
        "proc_len": int(len(y)),
    }

    return y.astype(np.float32), stats
