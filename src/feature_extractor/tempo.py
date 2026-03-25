import numpy as np
import librosa
from matplotlib.ticker import ScalarFormatter
# import matplotlib.pyplot as plt


class CyclicTempogram:
    def __init__(self, sr=44100, hop_length=512, win_length=384):
        self.sr = sr
        self.hop_length = hop_length
        self.win_length = win_length

    def extract(self, audio):
        # tính onset strength O(m)
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length)
        pad_len = self.win_length // 2
        onset_pad = np.pad(onset_env, (pad_len, pad_len), mode='constant')
        tempogram_cols = []
        for m in range(len(onset_env)):
            o_frame = onset_pad[m: m + self.win_length]
            o_frame = o_frame - np.mean(o_frame)
            # dùng cửa sổ Hanning để làm mịn 2 đầu audio
            o_frame = o_frame * np.hanning(self.win_length)
            # tính autocorrelation
            r_tau = np.correlate(
                o_frame, o_frame, mode="full")[
                self.win_length - 1:]
            # fourier Transform
            tc_f = np.fft.rfft(r_tau)
            magnitude = np.abs(tc_f)
            # bỏ tần số thấp, nhiễu
            magnitude[:3] = 0
            tempogram_cols.append(magnitude)
        tempogram_matrix = np.array(tempogram_cols).T
        return tempogram_matrix / np.max(tempogram_matrix)

    def shape(self, tempo):
        print(f"Shape of output: {tempo.shape}")

    def plot(self, feature_matrix, ax):
        bpm_freqs = np.arange(
            feature_matrix.shape[0]) * (self.sr * 60 / (self.hop_length * self.win_length))
        times = librosa.frames_to_time(
            np.arange(
                feature_matrix.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length)
        feature_matrix = feature_matrix[1:, :]
        bpm_freqs = bpm_freqs[1:]
        ax.pcolormesh(
            times,
            bpm_freqs,
            feature_matrix,
            cmap="coolwarm",
            shading="nearest")
        ax.set_ylabel("beats per minute (BPM) [-]", fontsize=11)
        ax.set_yscale("log", base=2)
        ax.set_yticks([16, 64, 256])
        ax.set_yticklabels(["16", "64", "256"], fontsize=10)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylim(16, 512)
