import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from dataloader import get_data_loaders, create_dataset
# import mlflow # nào xài mlflow thì uncomment z
import argparse
# from config import ESC_CLASSES (type: List[str]) - nào load config ông nhớ load cnay nha


class ModelTester:
    def __init__(self, model_path, data_dir, feature_extractor, batch_size=32, num_classes=50):
        self.model_path = model_path
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = None
        self.val_ds = None

    # Load model
    def load_model(self):
        print(f"[INFO] Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.summary()

    # Load val
    def load_data(self):
        print("[INFO] Loading data...")
        _, self.val_ds = get_data_loaders(
            data_dir=self.data_dir,
            feature_extractor=self.feature_extractor,
            batch_size=self.batch_size
        )

    # Run inference, return true labels & predictions 
    def predict(self):
        print("[INFO] Running inference...")
        all_preds = []
        all_labels = []

        for batch_data, batch_labels in self.val_ds:
            preds = self.model.predict(batch_data, verbose=0)
            all_preds.extend(np.argmax(preds, axis=1))
            all_labels.extend(batch_labels)

        return np.array(all_labels), np.array(all_preds)

    # Classification Report
    def print_classification_report(self, y_true, y_pred):
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        report = classification_report(
            y_true, y_pred,
            target_names=ESC50_CLASSES,
            digits=4
        )
        print(report)
        return report

    # confusion mat
    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(22, 18))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=ESC50_CLASSES,
            yticklabels=ESC50_CLASSES,
            linewidths=0.4,
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontsize=13)
        ax.set_ylabel("True Label", fontsize=13)
        ax.set_title("Confusion Matrix — ESC-50", fontsize=15)
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        # plt.savefig(save_path, dpi=150)
        # plt.close()
        # print(f"[INFO] Confusion matrix saved at {save_path}")

    # compare với paper benchmark
    def benchmark_comparison(self, y_true, y_pred, paper_target=76.5):
        correct = np.sum(y_true == y_pred)
        accuracy = correct / len(y_true) * 100

        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON")
        print("=" * 60)
        print(f"  Our Model Accuracy : {accuracy:.2f}%")
        print(f"  Paper Target       : {paper_target:.2f}%")
        gap = accuracy - paper_target
        if gap >= 0:
            print(f"  Result             : PASSED  (+{gap:.2f}%)")
        else:
            print(f"  Result             : BELOW TARGET  ({gap:.2f}%)")
        print("=" * 60)
        return accuracy

    # ghi log mfflow
    # def log_to_mlflow(self, accuracy, report, cm_path):
    #     mlflow.set_experiment("ESC50_Audio_Classification")
    #     with mlflow.start_run(run_name="test_evaluation"):
    #         mlflow.log_metric("test_accuracy", accuracy / 100)
    #         mlflow.log_metric("paper_target", 76.5 / 100)
    #         mlflow.log_text(report, "classification_report.txt")
    #         mlflow.log_artifact(cm_path)
    #     print("[INFO] Results logged to MLflow.")

    # chạy test
    # def run(self, save_cm_path="confusion_matrix.png", log_mlflow=True):
    #     self.load_model()
    #     self.load_data()

    #     y_true, y_pred = self.predict()

    #     report = self.print_classification_report(y_true, y_pred)
    #     self.plot_confusion_matrix(y_true, y_pred, save_path=save_cm_path)
    #     accuracy = self.benchmark_comparison(y_true, y_pred)

    #     if log_mlflow:
    #         self.log_to_mlflow(accuracy, report, save_cm_path)

    #     return accuracy


if __name__ == "__main__":
    pass
    # truyền theo command line, này cần ko ko biết nx
    # parser = argparse.ArgumentParser(description="Test ESC-50 Audio Classification Model")
    # parser.add_argument("--model_path", type=str, required=True,
    #                     help="")
    # parser.add_argument("--data_dir", type=str, required=True,
    #                     help="")
    # parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--no_mlflow", action="store_true",
    #                     help="Disable MLflow logging")
    # args = parser.parse_args()

    # Feature extractor
    # feature_extractor = create_dataset(
    #     feature_extracting=...,   # thay bằng object extractor (MelSpectrogram, MFCC, v.v.)
    #     sample_rate=44100,
    #     duration=5
    # )

    # tester = ModelTester(
    #     model_path=args.model_path,
    #     data_dir=args.data_dir,
    #     feature_extractor=feature_extractor,
    #     batch_size=args.batch_size
    # )
    # tester.run(log_mlflow=not args.no_mlflow)
