"""
CENS Chromagram Module

This module implements Chroma Energy Normalized Statistics (CENS) chromagrams as described in the paper.
CENS chromagrams are robust to dynamics and noise variations, making them suitable for large-scale
music structure analysis and retrieval. They are derived from CQT chromagrams with additional
energy normalization and smoothing.

Parameters:
    sr : int, optional
        Sampling rate (default 44100)
    hop_length : int, optional
        Hop length between frames (default 512)
    n_chroma : int, optional
        Number of chroma bins (default 12)
    n_octave : int, optional
        Number of octaves to consider (default 7)
    fmin : float, optional
        Minimum frequency (default librosa.note_to_hz('C1') ≈ 32.70 Hz)
    norm : float, optional
        Normalization factor for CQT (default 1)
    window : str, optional
        Window function (default 'hann')
    bins_per_octave : int, optional
        Number of bins per octave (default 12)
    win_len_smooth : int, optional
        Window length for temporal smoothing in frames (default 41)
    norm_smooth : str, optional
        Normalization type for smoothing (default 'mean')
    dtype : type, optional
        Output data type (default np.float32)

Returns:
    np.ndarray: 2D array of shape (n_chroma, time_frames) representing the CENS chromagram.
"""

import librosa


class CENSChromagram:
    def __init__(self, sr=44100, hop_length=512, n_chroma=12, n_octave=7,
                 fmin=librosa.note_to_hz('C1'), norm=1, window='hann',
                 bins_per_octave=12, win_len_smooth=41, norm_smooth='mean'):
        self.sr = sr
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.n_octave = n_octave
        self.fmin = fmin
        self.norm = norm
        self.window = window
        self.bins_per_octave = bins_per_octave
        self.win_len_smooth = win_len_smooth
        self.norm_smooth = norm_smooth

    def transform(self, audio_1d):
        cens = librosa.feature.chroma_cens(
            y=audio_1d,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
            n_octave=self.n_octave,
            fmin=self.fmin,
            norm=self.norm,
            window=self.window,
            bins_per_octave=self.bins_per_octave,
            win_len_smooth=self.win_len_smooth,
            norm_smooth=self.norm_smooth
        )
        return cens
