import numpy as np
import librosa

class CyclicTempogram:
    def __init__(self, sr=44100, hop_length=512, n_cyclic=12, tempo_min=40, tempo_max=240, method='fourier', window=384, onset_hop_length=512, onset_smooth=9):
        self.sr = sr
        self.hop_length = hop_length
        self.n_cyclic = n_cyclic
        self.tempo_min = tempo_min
        self.tempo_max = tempo_max
        self.method = method
        self.window = window
        self.onset_hop_length = onset_hop_length
        self.onset_smooth = onset_smooth

    def transform(self, audio_1d):
        onset_env = librosa.onset.onset_strength(
            y=audio_1d,
            sr=self.sr,
            hop_length=self.onset_hop_length,
            aggregate=np.median,
            smooth=self.onset_smooth
        )

        if self.method == 'fourier':
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
                win_length=self.window,
                mode='fourier',
                norm=None
            )
        else:
            tempogram = librosa.feature.tempogram(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
                win_length=self.window,
                mode='autocorrelation',
                norm=None
            )

        n_tempo = tempogram.shape[0]
        tempo_bins = librosa.tempo_frequencies(n_tempo, sr=self.sr, hop_length=self.hop_length)

        mask = (tempo_bins >= self.tempo_min) & (tempo_bins <= self.tempo_max)
        tempo_bins_masked = tempo_bins[mask]
        tempogram_masked = tempogram[mask, :]

        log2_tempo = np.log2(tempo_bins_masked / self.tempo_min)
        cyclic_idx = np.floor(log2_tempo * self.n_cyclic).astype(int) % self.n_cyclic

        cyclic_temp = np.zeros((self.n_cyclic, tempogram.shape[1]), dtype=np.float32)
        for i, idx in enumerate(cyclic_idx):
            cyclic_temp[idx, :] += tempogram_masked[i, :]

        max_val = cyclic_temp.max()
        if max_val > 0:
            cyclic_temp = cyclic_temp / max_val
        return cyclic_temp