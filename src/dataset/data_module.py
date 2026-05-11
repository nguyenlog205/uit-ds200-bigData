# src/dataset/data_module.py
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.config_loader import load_config


# -------------------- FeatureDataset--------------------
class FeatureDataset(Dataset):
    """Dataset for pre-extracted .npy feature files using metadata CSV."""
    def __init__(
        self,
        metadata: pd.DataFrame, 
        feature_name: str,
        augment: bool = False
    ):
        self.metadata = metadata[metadata['feature_name'] == feature_name].reset_index(drop=True)
        self.augment = augment
        if len(self.metadata) == 0:
            raise ValueError(f"No samples found for feature_name='{feature_name}'")
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        feat_path = row['feature_path']
        label = int(row['target'])

        feature = np.load(feat_path)
        feature_tensor = torch.from_numpy(feature).float()

        if feature_tensor.ndim == 2:
            feature_tensor = feature_tensor.unsqueeze(0)
        elif feature_tensor.ndim == 1:
            feature_tensor = feature_tensor.unsqueeze(0)

        if self.augment:
            feature_tensor = self._apply_augment(feature_tensor)
        return feature_tensor, label

    def _apply_augment(self, x):
        _, H, W = x.shape
        max_t_mask = max(1, W // 20)
        t_mask = torch.randint(1, max_t_mask + 1, (1,)).item()
        t_start = torch.randint(0, max(1, W - t_mask), (1,)).item()
        x[:, :, t_start:t_start + t_mask] = 0.0

        max_f_mask = max(1, H // 20)
        f_mask = torch.randint(1, max_f_mask + 1, (1,)).item()
        f_start = torch.randint(0, max(1, H - f_mask), (1,)).item()
        x[:, f_start:f_start + f_mask, :] = 0.0
        return x

# -------------------- FeatureDataModule --------------------
class FeatureDataModule:
    def __init__(
        self,
        feature_name: str,
        experiment_configuration_path: str = 'configs/experiments/cnn_models.yml',
    ):
        self.config = load_config(experiment_configuration_path)
        self.feature_name = feature_name

        feature_root = Path(self.config['dataset']['feature_root'])
        self.metadata_path = {
            'train': feature_root / 'train.csv',
            'dev': feature_root / 'val.csv',
            'test': feature_root / 'test.csv',
        }
        # print(feature_root / 'dev.csv')

        self.batch_size = self.config['training']['batch_size']
        self.num_workers = self.config['dataset']['dataloader']['num_workers']

        # Load metadata ngay trong __init__
        self.metadata = {
            'train': pd.read_csv(self.metadata_path['train']),
            'dev': pd.read_csv(self.metadata_path['dev']),
            'test': pd.read_csv(self.metadata_path['test']),
        }

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.is_augment = self.config['others']['augment']

    def setup(self):
        """Tạo dataset từ metadata đã load."""
        self.train_dataset = FeatureDataset(self.metadata['train'], self.feature_name, augment=self.is_augment)
        self.val_dataset = FeatureDataset(self.metadata['dev'], self.feature_name, augment= False)
        self.test_dataset = FeatureDataset(self.metadata['test'], self.feature_name, augment=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


# ---------------------------------------
if __name__ == '__main__':
    dm = FeatureDataModule(feature_name='melspectrogram')
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")
    sample, label = dm.train_dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")