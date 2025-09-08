from typing import Optional

import numpy as np
import python_speech_features as psf
from pydub import AudioSegment

from .segmentation import audiosegment_to_np_array


def compute_mfcc_from_segment(segment: AudioSegment) -> np.ndarray:
    """Compute MFCC features from an audio segment.

    Uses parameters aligned with the provided prototype and applies
    weighting to MFCC 2..7 (inclusive) to emphasize informative components.
    """
    sample_rate = segment.frame_rate
    audio_data = audiosegment_to_np_array(segment)

    mfcc_features = psf.mfcc(
        signal=audio_data,
        samplerate=sample_rate,
        numcep=24,
        nfilt=48,
        nfft=2048,
    )

    weights = np.ones(mfcc_features.shape[1])
    weights[2:8] *= 1.5
    mfcc_features = mfcc_features * weights

    return mfcc_features


def compute_mfcc_from_file(audio_file: str) -> np.ndarray:
    """Load an audio file and compute MFCC features for the entire audio."""
    segment = AudioSegment.from_file(audio_file)
    return compute_mfcc_from_segment(segment)


def quantize_vector(values: np.ndarray, decimals: int = 1) -> np.ndarray:
    """Quantize values by rounding to a fixed number of decimals."""
    return np.round(values, decimals=decimals)


def serialize_fingerprint(feature_matrix: np.ndarray, decimals: int = 1) -> str:
    """Serialize an MFCC matrix into a compact string fingerprint.

    Implements: F = Serialize(Q((1/T) sum_t (M_t ⊙ W)))
    where weighting W is already applied in compute_mfcc_from_segment.
    """
    averaged = np.mean(feature_matrix, axis=0)
    quantized = quantize_vector(averaged, decimals=decimals)
    return ",".join(map(str, quantized))


def mfcc_to_string(mfcc_array: np.ndarray, decimals: int = 1) -> str:
    """Compatibility alias mirroring prototype naming."""
    return serialize_fingerprint(mfcc_array, decimals=decimals)


def fingerprint_from_segment(segment: AudioSegment, decimals: int = 1) -> str:
    """Compute weighted MFCC and return serialized fingerprint string."""
    mfcc = compute_mfcc_from_segment(segment)
    return serialize_fingerprint(mfcc, decimals=decimals)


def fingerprint_from_file(audio_file: str, decimals: int = 1) -> str:
    """Compute fingerprint directly from an audio file."""
    mfcc = compute_mfcc_from_file(audio_file)
    return serialize_fingerprint(mfcc, decimals=decimals)


