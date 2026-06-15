import torch
import torchaudio


class LogMelExtractor(torch.nn.Module):
    def __init__(
        self,
        sr=4000,
        n_fft=1024,
        win_length=400,
        hop_length=160,
        n_mels=128,
    ):
        super().__init__()

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, waveform):
        logmel = self.to_db(self.mel(waveform))

        mean = logmel.mean(dim=(-2, -1), keepdim=True)
        std = logmel.std(dim=(-2, -1), keepdim=True) + 1e-6

        return (logmel - mean) / std
