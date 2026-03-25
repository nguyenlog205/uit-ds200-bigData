import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
"""
Cyclic Tempogram Module

Module encodes the tempo of an audio signal over time by analyzing rythmic information and mapping it using a Fourier transform

Parameters:
    sr: int, optinal
        Sampling rate (default: 44100)
    hop_length: int, optinal
        hop_length between frames (default: 512)
    win_length: int, optional 
        win_length for tepogram in frames (default: 384)

Returns:
    np.ndarray: 2D array of shape (win_length // 2 + 1, time_frames) representing the cyclic tempogram.
"""
class CyclicTempogram:
    def __init__(self, sr=44100, hop_length=512, win_length=384):
        self.sr = sr 
        self.hop_length = hop_length
        self.win_length = win_length

    def extract(self, audio):
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr, hop_length=self.hop_length)
        tempo = librosa.feature.fourier_tempogram(
            onset_envelope = onset_env,
            sr = self.sr,
            hop_length = self.hop_length,
            win_length = self.win_length
        )
        magnitude = np.abs(tempo)
        return magnitude
    
    def shape(self, tempo):
        print(f"Shape of output: {tempo.shape}")
    
    def plot(self, tempo):
        """
        x_coords và y_coords dùng để truyền times, bpm_freqs vào --> mapping từng pixel theo times và bpm_freqs
        Hiển thị ma trận STFT_chroma.
        - Trục x: Thời gian (Time).
        - Trục y: tần số (BPM).
        """
        tempo = tempo[1: ,:]
        bpm_freqs = librosa.fourier_tempo_frequencies(sr=self.sr, hop_length=self.hop_length, win_length=self.win_length)
        times = librosa.times_like(tempo, sr=self.sr, hop_length=self.hop_length)
        bpm_freqs = bpm_freqs[1:]
        img = librosa.display.specshow(
            tempo,
            sr = self.sr,
            hop_length = self.hop_length,
            x_coords = times,
            y_coords = bpm_freqs,
            x_axis = "time",
            cmap = "coolwarm"
        )
        ax = plt.gca()
        ax.set_yscale("log", base=2)
        ax.set_ylim(16, 512)
        ax.set_yticks([16, 64, 256])
        ax.set_yticklabels(["16", "64", "256"])
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylabel("beats per minute (BPM)[-]", fontsize=11)
        ax.set_xlabel("time[s]", fontsize=11)
        plt.title("Cyclic Tempogram")
        plt.tight_layout()
        plt.show()