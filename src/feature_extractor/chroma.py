import numpy as np
import librosa
import librosa.display
# import matplotlib.pyplot as plt

"""
Module use short-time Fourier transform algorithms for audio signal processing to extract the frequency information of audio over time.
Module analyze harmonic and tonal content, mapping spectral energy to 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)

Parameters:
    sr : int, optional
        Sampling rate (default: 44100)
    n_fft : int, optional
        Length of the FFT window (default: 2048)
    hop_length : int, optional
        Number of samples between successive frames (default: 512)
    n_chroma : int, optional
        Number of chroma bins to extract (default: 12)
    window : str, optional
        Window function to apply (default: 'hann')

Returns:
    np.ndarray: 2D array of shape (n_chroma, time_frames) representing the STFT chromagram.
"""
class STFTChromagram:
    def __init__(self, sr=44100, hop_length=512, n_fft=2048, n_chroma=12, window ="hann"):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma
        self.window = window

    def extract(self, audio):
        chroma = librosa.feature.chroma_stft(
            y = audio,
            sr = self.sr,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            n_chroma = self.n_chroma,
            window = self.window
        )
        return chroma
    
    def shape(self, STFT):
        print(f"Shape of output: {STFT.shape}")

    def plot(self, chroma):
        """
        Hiển thị ma trận STFT_chroma.
        - Trục x: Thời gian (Time).
        - Trục y: lớp theo giá trị tần số (pitch class).
        """
        feature_matrix_3x = np.tile(chroma, (3, 1))
        img = librosa.display.specshow(
            feature_matrix_3x, 
            x_axis = "time", 
            sr = self.sr, 
            hop_length = self.hop_length, 
            cmap = "magma"
        )
        plt.ylabel("tone pitch class [-]", fontsize=11)
        y_ticks_pos = [0.5, 4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5]
        y_ticks_labels = ['C', 'E', 'G#', 'C', 'E', 'G#', 'C', 'E', 'G#']
        plt.yticks(y_ticks_pos, y_ticks_labels, fontsize=11)
        plt.ylim(0, 36)
        plt.title("STFT Chromagram (3 Octaves)")
        plt.tight_layout()
        plt.show()
