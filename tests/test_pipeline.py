"""
Unit tests for feature extraction modules and CNN model architecture.
"""

import numpy as np
import pytest

# Existing feature extractors
from src.dataset.feature_extractor.melspectrogram import MelScaleSpectrogram
from src.dataset.feature_extractor.mfcc import MFCC
from src.dataset.feature_extractor.tempo import CyclicTempogram          # cyclic tempogram
from src.dataset.feature_extractor.chroma import STFTChromagram          # STFT chromagram
from src.dataset.feature_extractor.ctq_chromagram import CQTChromagram   # CQT chromagram (filename typo)
from src.dataset.feature_extractor.cens_chromagram import CENSChromagram

AUDIO_LEN = 220500
MOCK_AUDIO = np.random.randn(AUDIO_LEN).astype(np.float32)

# List of all six feature modules (matching the paper)
FEATURE_MODULES = [
    (MelScaleSpectrogram, {}),
    (MFCC, {}),
    (CyclicTempogram, {}),
    (STFTChromagram, {}),
    (CQTChromagram, {}),
    (CENSChromagram, {}),
]

# ----------------------------------------------------------------------
# Feature Tests
# ----------------------------------------------------------------------


@pytest.mark.parametrize("feature_class, kwargs", FEATURE_MODULES)
def test_feature_returns_2d_matrix(feature_class, kwargs):
    """
    Verify that each feature module returns a 2D matrix (height, time).
    """
    extractor = feature_class(**kwargs)
    result = extractor.transform(MOCK_AUDIO)
    assert isinstance(result, np.ndarray), f"{feature_class.__name__} did not return a numpy array"
    assert result.ndim == 2, f"{feature_class.__name__} returned {result.ndim}D, expected 2D"
    # Also ensure height > 0 and time > 0
    assert result.shape[0] > 0, "Height dimension is zero"
    assert result.shape[1] > 0, "Time dimension is zero"

# ----------------------------------------------------------------------
# Model Architecture
# ----------------------------------------------------------------------

def test_cnn_architecture():
    import yaml
    with open('configs/model.yml') as f:
        configs = yaml.safe_load(f)['cnn']
    
    CNNArchitect = pytest.importorskip("src.models.cnn").CNNArchitect

    architect = CNNArchitect(configs)
    model = architect.build_model()

    # Check condition
    # <This code was composed by Long Nguyen Hoang>


