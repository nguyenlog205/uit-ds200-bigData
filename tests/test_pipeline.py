"""
Unit tests for feature extraction modules and CNN model architecture.
"""

import numpy as np
import pytest

# Existing feature extractors
from src.feature_extractor.melspectrogram import MelScaleSpectrogram
from src.feature_extractor.mfcc import MFCC
from src.feature_extractor.tempo import CyclicTempogram          # cyclic tempogram
from src.feature_extractor.chroma import STFTChromagram          # STFT chromagram
from src.feature_extractor.ctq_chromagram import CQTChromagram   # CQT chromagram (filename typo)
from src.feature_extractor.cens_chromagram import CENSChromagram

# TODO: create model_factory.py and import from it
# from src.models.model_factory import ModelFactory, CNNArchitect

# Mock audio: 5 seconds at 44.1 kHz -> 220500 samples
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
# Model Architecture Test (disabled until model_factory is implemented)
# ----------------------------------------------------------------------

# def test_cnn_architect_has_13_layers():
#     """
#     Verify that CNNArchitect built model has exactly 13 layers,
#     as per the paper's architecture description.
#     """
#     architect = CNNArchitect(input_shape=(128, None, 1), num_classes=50)
#     model = architect.build_model()
#     layer_count = len(model.layers)
#     assert 12 <= layer_count <= 15, f"Unexpected number of layers: {layer_count}"
#     layer_types = [type(layer).__name__ for layer in model.layers]
#     assert 'Conv2D' in layer_types, "No Conv2D layers found"
#     assert 'MaxPooling2D' in layer_types, "No MaxPooling2D layers found"
#     assert 'Dense' in layer_types, "No Dense layers found"

# ----------------------------------------------------------------------
# Input/Output Shape Test (disabled)
# ----------------------------------------------------------------------

# @pytest.mark.parametrize("feature_name, model_type, input_shape", [
#     ('mel_spectrogram', 'cnn', (128, None, 1)),
#     ('mfcc', 'cnn', (20, None, 1)),
#     ('cyclic_tempogram', 'cnn', (12, None, 1)),
#     ('stft_chromagram', 'cnn', (12, None, 1)),
#     ('cqt_chromagram', 'cnn', (12, None, 1)),
#     ('cens_chromagram', 'cnn', (12, None, 1)),
# ])
# def test_model_input_shape_matches_feature_output(
#         feature_name, model_type, input_shape):
#     # ... test code
#     pass