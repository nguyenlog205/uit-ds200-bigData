import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Dict, List
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm  # Thêm thư viện tqdm

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training parameters
        self.epochs = config.get('epochs', 50)
        self.lr = config.get('learning_rate', 0.001)
        self.optimizer_name = config.get('optimizer', 'Adam')
        self.weight_decay = config.get('weight_decay', 0.0)
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.early_stop_patience = config.get('early_stop_patience', 10)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self._init_optimizer()

        # History container
        self.history: Dict[str, List[float]] = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

    def _init_optimizer(self):
        if self.optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        best_val_loss = float('inf')
        patience_counter = 0

        # Thanh tiến trình cho tổng số epoch
        epoch_pbar = tqdm(range(self.epochs), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            # Training (có thanh tiến trình cho từng batch)
            train_loss, train_acc, train_f1 = self._train_one_epoch(train_loader, epoch)

            # Validation
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)

            # Lưu history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)

            # Cập nhật thông tin trên thanh tiến trình epoch
            epoch_pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_loss': f"{val_loss:.4f}",
                'val_acc': f"{val_acc:.4f}"
            })

            # Ghi log chi tiết (dùng tqdm.write để không phá vỡ thanh tiến trình)
            tqdm.write(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
            )

            # Early stopping & checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    tqdm.write(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Lưu history
        self._save_history()
        tqdm.write("Training finished.")

    def _train_one_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Thanh tiến trình cho từng batch trong một epoch
        batch_pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False, unit="batch")

        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Cập nhật loss trên thanh batch (tùy chọn)
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        return avg_loss, acc, f1

    def _evaluate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        # Thanh tiến trình cho validation (có thể để leave=False để ẩn sau khi xong)
        val_pbar = tqdm(loader, desc="Validation", leave=False, unit="batch")

        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        return avg_loss, acc, f1

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'epoch_{epoch+1}.pth'
        torch.save(checkpoint, path)
        tqdm.write(f"Checkpoint saved to {path}")

    def _save_history(self):
        """Save training history to CSV file."""
        df = pd.DataFrame(self.history)
        csv_path = self.save_dir / 'history.csv'
        df.to_csv(csv_path, index=False)
        tqdm.write(f"Training history saved to {csv_path}")