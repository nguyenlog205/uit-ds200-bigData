import numpy as np
import librosa
import librosa.display
# import matplotlib.pyplot as plt


class STFTChromagram:
    def __init__(self, sr=44100, hop_length=512, n_fft=2048, n_chroma=12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.window = np.hanning(n_fft)

    def extract(self, audio):
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        stft_matrix = np.empty((self.n_fft // 2 + 1, n_frames), dtype=complex)
        # biến đổi Fourier theo STFT theo từng chỉ số frame
        for m in range(n_frames):
            start = m * self.hop_length
            # sử dụng cửa sổ để làm mịn 2 đầu audio
            x_windowed = audio[start:start + self.n_fft] * self.window
            stft_matrix[:, m] = np.fft.rfft(x_windowed)
        # tính phổ công suất
        power_spec = np.abs(stft_matrix)**2
        # gom tần số theo lớp
        chroma_filters = librosa.filters.chroma(
            sr=self.sr, n_fft=self.n_fft, n_chroma=self.n_chroma)
        chromagram = np.dot(chroma_filters, power_spec)
        chromagram = chromagram / np.max(chromagram)
        return chromagram

    def shape(self, STFT):
        print(f"Shape of output: {STFT.shape}")

    def plot(self, feature_matrix, ax):
        # nhân ma trận 12 lớp 3 lần
        '''
        feature_matrix_3x = np.tile(feature_matrix, (3, 1))
        img = librosa.display.specshow(
            feature_matrix_3x,
            x_axis="time",
            sr=self.sr,
            hop_length=self.hop_length,
            ax=ax,
            cmap="magma")
        '''
        ax.set_ylabel("tone pitch class [-]", fontsize=11)
        y_ticks_pos = [0.5, 4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5]
        y_ticks_labels = ['C', 'E', 'G#', 'C', 'E', 'G#', 'C', 'E', 'G#']
        ax.set_yticks(y_ticks_pos)
        ax.set_yticklabels(y_ticks_labels, fontsize=11)
        ax.set_ylim(0, 36)
