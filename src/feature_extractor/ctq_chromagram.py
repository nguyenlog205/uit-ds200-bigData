"""
CQT Chromagram Module

This module implements Constant-Q Transform (CQT) chromagram extraction as described in the paper.
CQT chromagrams use a logarithmic frequency resolution, providing higher frequency resolution at lower pitches,
making them suitable for harmonic analysis and chord recognition.

Parameters:
    sr : int, optional
        Sampling rate of the audio signal (default 44100)
    hop_length : int, optional
        Number of samples between successive frames (default 512)
    n_chroma : int, optional
        Number of chroma bins (12 for standard chromatic scale) (default 12)
    n_bins : int, optional
        Number of frequency bins in the CQT (default 84, i.e., 7 octaves)
    bins_per_octave : int, optional
        Number of bins per octave (default 12)
    fmin : float, optional
        Minimum frequency (default librosa.note_to_hz('C1') ≈ 32.70 Hz)
    norm : float, optional
        Normalization factor for the CQT (default 1)
    window : str, optional
        Window function (default 'hann')
    dtype : type, optional
        Data type for the output (default np.float32)

Returns:
    np.ndarray: 2D array of shape (n_chroma, time_frames) representing the CQT chromagram.
"""

import librosa


class CQTChromagram:
    def __init__(self, sr=44100, hop_length=512, n_chroma=12,
                 bins_per_octave=12, fmin=librosa.note_to_hz('C1'),
                 norm=1):
        self.sr = sr
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.bins_per_octave = bins_per_octave
        self.fmin = fmin
        self.norm = norm

    def transform(self, audio_1d):
        chroma = librosa.feature.chroma_cqt(
            y=audio_1d,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
            bins_per_octave=self.bins_per_octave,
            fmin=self.fmin,
            norm=self.norm
        )
        return chroma
