import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
# import pandas as pd


class MFCC:
    def __init__(
            self,
            sr=44100,
            n_mfcc=20,
            n_fft=2048,
            hop_length=512,
            n_mels=128):
        """
        MFCC constructor
        :param n_mfcc: Số lượng hệ số MFCC cần lấy (từ 13-40)
        - power mặc định = 2.0 khi hàm transform melspectrogram được gọi trong method mfcc()
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def transform(self, audio_1d) -> np.ndarray:
        """
        Extract 1D MFCC matrix directly from audio (mel-spectrogram computed in mfcc() method)
        """
        audio_1d = audio_1d.astype(np.float32)
        mfccs = librosa.feature.mfcc(
            y=audio_1d,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        return mfccs  # 2D matrix (n_mfccs x n_frames)

    def display(self, mfccs: np.ndarray, title="MFCC"):
        """
        Hiển thị ma trận MFCC.
        - Trục y: Tần số.
        - Trục x: Thời gian (Time).
        - Màu sắc: Giá trị của hệ số (biên độ).
        """
        plt.figure(figsize=(10, 4))

        img = librosa.display.specshow(
            mfccs,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmax=self.sr / 2
        )

        plt.colorbar(img)
        plt.title(title)
        plt.ylabel('Hz')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mfcc_extractor = MFCC()
    path = "/kaggle/input/datasets/mmoreaux/environmental-sound-classification-50/audio/audio/1-100032-A-0.wav"
    y, sr = librosa.load(path, sr=44100)
    mfcc_data = mfcc_extractor.transform(y)
    # (20, 431): 20 -> n_mfcc, 431 -> n frames
    print(f"MFCC shape: {mfcc_data.shape}")
