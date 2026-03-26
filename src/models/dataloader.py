import numpy as np
import tensorflow as tf
import os
from glob import glob
from sklearn.model_selection import train_test_split
import librosa

class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, feature_extracting, batch_size, sample_rate, duration, shuffle, **kwargs):
        # kế thừa hàm khởi tạo và tất cả tham số của lớp cha keras
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.feature_extracting = feature_extracting
        self.sample_rate = sample_rate
        self.duration = duration
        self.shuffle = shuffle
        self.n_samples = len(file_paths)

    def __getitem__(self, key):
        start = key * self.batch_size
        stop = min((key + 1) * self.batch_size, self.n_samples)
        batch_paths = self.file_paths[start:stop]
        batch_labels = self.labels[start:stop]
        
        data = []
        for path in batch_paths:
            feature = self.feature_extracting(path)
            data.append(feature)
        data = np.array(data)
        batch_labels = np.array(batch_labels)
        return data, batch_labels
    
    def __len__(self):
        # .ceil(): lấy tròn số lên 
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            index_list = np.arange(self.n_samples)
            np.random.shuffle(index_list)
            file_paths_tmp = []
            labels_tmp = []
            for i in index_list:
                file_paths_tmp.append(self.file_paths[i])
                labels_tmp.append(self.labels[i])
            self.file_paths = file_paths_tmp
            self.labels = labels_tmp

def get_data_loaders(data_dir, feature_extractor, batch_size=32, sample_rate=44100, test_size=0.2):
    all_files = glob(os.path.join(data_dir, "*.wav"))

    all_labels =[]
    for file in all_files:
        all_labels.append(int(os.path.basename(file).split("-")[-1].replace(".wav", "")))

    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=42
    )
    train = DataLoader(train_files, train_labels, feature_extractor, batch_size, sample_rate, duration=5, shuffle=True)
    val = DataLoader(val_files, val_labels, feature_extractor, batch_size, sample_rate, duration=5, shuffle=False)
    return train, val

def create_dataset(feature_extracting, sample_rate, duration):
    def processor(file_path):
        audio_array, _ = librosa.load(file_path, sr=sample_rate, duration=duration)
        feature_matrix = feature_extracting.extract(audio_array)
        # Matrix 2D --> thêm chiều số channel mặc định là 1 để cnn chạy (height, width, 1)
        feature_cnn = np.expand_dims(feature_matrix, axis=-1)
        return feature_cnn
    return processor