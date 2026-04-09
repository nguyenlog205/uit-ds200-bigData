import numpy as np
import librosa
import librosa.display

class STFTChromagram:
    def __init__(self, sr=44100, hop_length=512, n_fft=2048, n_chroma=12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_chroma = n_chroma

    def transform(self, audio):
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        return chroma

    def shape(self, feature):
        print(f"Shape of output: {feature.shape}")

    def plot(self, feature_matrix, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        # Optional: tile for 3 octaves display
        feature_matrix_3x = np.tile(feature_matrix, (3, 1))
        img = librosa.display.specshow(
            feature_matrix_3x,
            x_axis="time",
            sr=self.sr,
            hop_length=self.hop_length,
            ax=ax,
            cmap="magma"
        )
        ax.set_ylabel("tone pitch class [-]", fontsize=11)
        y_ticks_pos = [0.5, 4.5, 8.5, 12.5, 16.5, 20.5, 24.5, 28.5, 32.5]
        y_ticks_labels = ['C', 'E', 'G#', 'C', 'E', 'G#', 'C', 'E', 'G#']
        ax.set_yticks(y_ticks_pos)
        ax.set_yticklabels(y_ticks_labels, fontsize=11)
        ax.set_ylim(0, 36)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("STFT Chromagram (3 Octaves)")
        plt.tight_layout()
        plt.show()
