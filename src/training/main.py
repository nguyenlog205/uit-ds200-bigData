import argparse
import logging
from pathlib import Path
import torch

from src.utils.config_loader import load_config
from src.models.factory import ModelFactory
from src.dataset.data_module import FeatureDataModule
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/experiments/cnn_models.yml',
                        help='Path to experiment configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")

    # Extract feature list
    feature_details = config['dataset']['feature_details']
    feature_names = feature_details['name']
    input_dims = feature_details['input_dim']

    # Ensure model config has correct num_classes
    model_cfg = config['model']
    model_cfg['num_classes'] = config['dataset']['num_classes']

    # Training base config
    training_cfg = config['training']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Loop over each feature
    for fname in feature_names:
        logger.info(f"=== Starting experiment for feature: {fname} ===")

        # 1. Prepare DataModule
        datamodule = FeatureDataModule(
            feature_name=fname,
            experiment_configuration_path=args.config
        )
        datamodule.setup()
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # 2. Create model (adjust input_channels if needed; CNN uses 1)
        model = ModelFactory.create_model(model_cfg)

        # 3. Override save_dir for this feature
        feat_save_dir = Path(training_cfg['save_dir']) / f"{model_cfg['name']}_{fname}"
        feat_training_cfg = training_cfg.copy()
        feat_training_cfg['save_dir'] = str(feat_save_dir)

        # 4. Train
        trainer = Trainer(model=model, config=feat_training_cfg, device=device)
        trainer.fit(train_loader, val_loader)

        logger.info(f"=== Finished experiment for feature: {fname} ===\n")

if __name__ == '__main__':
    main()
    # python -m src.training.train --config configs/experiments/cnn_models.yml