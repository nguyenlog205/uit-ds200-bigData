import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


class MelScaleSpectrogram:
    def __init__(self, sr=44100, to_db=True, n_fft=2048, hop_length=512, n_mels=128, power=2.0):
        """
        Khởi tạo module Mel-spectrogram với các siêu tham số.
        :param sr: Sampling rate (mặc định 44.1kHz)
        :param n_fft: Độ dài của cửa sổ FFT (độ phân giải tần số)
        :param hop_length: Khoảng cách giữa các cửa sổ (độ phân giải thời gian)
        :param n_mels: Số lượng bộ lọc Mel
        :param power: lấy biên độ (1.0) hay lấy phổ năng lượng (2.0), mặc định 2.0
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.power = power
        self.to_db = to_db

    def transform(self, audio_1d, to_db=None) -> np.ndarray:
        """
        Trích xuất mel-scaled spectrogram
        :param to_db: Nếu truyền True/False sẽ ghi đè cấu hình lúc khởi tạo
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio_1d,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power
        )  # output shape: (n_mels, n_frames)

        should_convert = to_db if to_db is not None else self.to_db
        if should_convert:
            return librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec

    def print_shape(self, mel_spec):
        print(f"Shape of output: {mel_spec.shape}")

    def display(self, mel_spec: np.ndarray, title="Mel-scaled Spectrogram"):
        """
        Hiển thị mel-spectrogram
        - Trục y: tần số (mel scaled)
        - Trục x: thời gian
        - Màu sáng hơn thì năng lượng cao hơn
        :param mel_spec: truyền vào nên là kết quả từ transform().
        """
        plt.figure(figsize=(10, 4))

        # Vẽ spectrogram
        img = librosa.display.specshow(
            mel_spec,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmax=self.sr / 2
        )

        plt.colorbar(img, format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mel_extractor = MelScaleSpectrogram(to_db=True)
    path = "/kaggle/input/datasets/mmoreaux/environmental-sound-classification-50/audio/audio/1-100032-A-0.wav"
    y, sr = librosa.load(path, sr=44100)
    mel_data = mel_extractor.transform(y)
    print(f"Mel shape: {mel_data.shape}")