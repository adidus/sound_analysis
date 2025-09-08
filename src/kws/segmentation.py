import os
from typing import List

import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence


def normalize_volume(audio: AudioSegment, target_dBFS: float = -20.0) -> AudioSegment:
    """Normalize audio to the target dBFS level.

    Args:
        audio: Input AudioSegment.
        target_dBFS: Desired loudness level.

    Returns:
        AudioSegment normalized to target dBFS.
    """
    change_in_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_in_dBFS)


def split_audio_by_silence(
    audio_file: str,
    min_silence_len: int = 500,
    silence_thresh: int = -35,
    keep_silence: int = 100,
) -> List[AudioSegment]:
    """Split audio file into segments based on silence.

    Args:
        audio_file: Path to audio file.
        min_silence_len: Minimum silence length in ms.
        silence_thresh: Silence threshold in dBFS.
        keep_silence: Padding of silence to keep around segments in ms.

    Returns:
        List of AudioSegment chunks.
    """
    audio = AudioSegment.from_file(audio_file)
    audio = normalize_volume(audio)
    audio_chunks = split_on_silence(
        audio_segment=audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence,
    )
    print(f"Total segments detected: {len(audio_chunks)}")
    return audio_chunks


def save_audio_segments(audio_file: str, audio_segments: List[AudioSegment]) -> None:
    """Save segmented audio chunks next to the source file as WAVs."""
    output_dir = os.path.splitext(audio_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    for i, segment in enumerate(audio_segments):
        segment_filename = f"{output_dir}/segment_{i}.wav"
        segment.export(segment_filename, format="wav")
        print(f"Segment {i} saved as {segment_filename}")


def audiosegment_to_np_array(segment: AudioSegment) -> np.ndarray:
    """Convert an AudioSegment to a NumPy int16 array."""
    samples = segment.get_array_of_samples()
    audio_data = np.array(samples, dtype=np.int16)
    return audio_data


