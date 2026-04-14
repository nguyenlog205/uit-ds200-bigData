# src/models/factory.py
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any

from src.utils.config_loader import load_config

from .cnn import CNN
from .ast import AST
from .capsule_network import CapsuleNetwork
from .linear_svm import LinearSVM
from .knn import KNNClassifier


class ModelFactory:
    """
    Factory for creating model instances from configuration dictionaries or YAML files.
    Supports CNN, AST, CapsuleNetwork, LinearSVM, and KNNClassifier.
    """

    @staticmethod
    def create_model(
        config: Union[Dict[str, Any], str, Path],
        weights_path: Optional[Union[str, Path]] = None,
        knn_data_path: Optional[Dict[str, Union[str, Path]]] = None,
    ) -> torch.nn.Module:
        """
        Instantiate a model from configuration.

        Args:
            config: Model configuration as a dict or path to a YAML file.
            weights_path: Optional path to a .pth file containing state_dict.
                          Ignored for KNNClassifier.
            knn_data_path: Optional dict with keys 'x' and 'y' pointing to .npy files
                           for KNNClassifier training data.

        Returns:
            Instantiated PyTorch model.
        """
        # =====================================================
        # --- Load config if a path is provided ---
        # =====================================================
        if isinstance(config, (str, Path)):
            config = load_config(config)


        # =====================================================
        # --- Extract parameters ---
        # =====================================================
        # Determine model type
        model_type = config.get("type") or config.get("name")
        if model_type is None:
            raise ValueError("Config must contain either 'type' or 'name' key.")

        # Extract other parameters
        if "parameters" in config:
            params = config["parameters"]
        else:
            params = {k: v for k, v in config.items() if k not in ("type", "name")}


        # =====================================================
        # --- Create model instance ---
        # =====================================================
        if model_type == "cnn":
            model = CNN(
                num_classes=params.get("num_classes", 50),
                input_channels=params.get("input_channels", 1),
            )
        elif model_type == "ast":
            model = AST(
                num_classes=params.get("num_classes", 50),
                input_fdim=params.get("input_fdim", 128),
                input_tdim=params.get("input_tdim", 512),
                patch_size=params.get("patch_size", 16),
                embed_dim=params.get("embed_dim", 768),
                num_heads=params.get("num_heads", 12),
                num_layers=params.get("num_layers", 12),
            )
        elif model_type == "capsule_network":
            model = CapsuleNetwork(
                num_classes=params.get("num_classes", 50),
                input_channels=params.get("input_channels", 1),
            )
        elif model_type == "linear_svm":
            model = LinearSVM(
                input_dim=params.get("input_dim", 40),
                num_classes=params.get("num_classes", 50),
                C=params.get("C", 1.0),
            )
        elif model_type == "KNNClassifier":
            k = params.get("k", 5)
            model = KNNClassifier(k=k)
            if knn_data_path is not None:
                x_train = torch.from_numpy(np.load(knn_data_path["x"]))
                y_train = torch.from_numpy(np.load(knn_data_path["y"])).long()
                model.fit(x_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if weights_path is not None and model_type != "KNNClassifier":
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)

        return model
    
if __name__ == '__main__':
    # python -m src.models.factory --model cnn 
    # python -m src.models.factory --model ast 
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type (CNN, AST, CapsuleNetwork, LinearSVM, KNNClassifier)")
    parser.add_argument("--weights", type=str, default=None, help="Optional path to weights file (.pth)")
    args = parser.parse_args()
    config_dict = {"type": args.model, "parameters": {"num_classes": 50}}
    
    model = ModelFactory.create_model(config_dict)
    print(model)

    # model = ModelFactory.create_model("cnn.yaml", weights_path="cnn_weights.pth")

    
    # knn = ModelFactory.create_model(
    #     "knn.yaml",
    #     knn_data_path={
    #         "x": "knn_xtrain.npy", 
    #         "y": "knn_ytrain.npy"
    #     }
    # )