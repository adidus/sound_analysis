import os
from typing import Dict, Tuple

from pydub import AudioSegment

from .features import compute_mfcc_from_segment, mfcc_to_string


def levenshtein_distance(s1: str, s2: str) -> int:
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[len(s1)][len(s2)]


def load_keyword_dictionary(directory: str) -> Dict[str, str]:
    """Build dictionary: keyword -> fingerprint string from reference audio files."""
    keyword_dict: Dict[str, str] = {}

    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}")
        return keyword_dict

    for filename in os.listdir(directory):
        if filename.lower().endswith((".mp3", ".wav", ".flac", ".ogg")):
            keyword = os.path.splitext(filename)[0]
            path = os.path.join(directory, filename)
            try:
                audio = AudioSegment.from_file(path)
                mfcc = compute_mfcc_from_segment(audio)
                keyword_dict[keyword] = mfcc_to_string(mfcc)
                print(f"Loaded keyword: {keyword}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to process {filename}: {exc}")

    return keyword_dict


def recognize_keyword(segment: AudioSegment, keyword_dict: Dict[str, str], distance_threshold: int = 3000) -> Tuple[str, int]:
    """Recognize keyword via Levenshtein distance to reference fingerprints.

    Returns the best keyword and the corresponding distance.
    """
    mfcc_segment = compute_mfcc_from_segment(segment)
    seg_str = mfcc_to_string(mfcc_segment)

    best_keyword = "no_keyword"
    min_distance = float("inf")

    for keyword, ref_str in keyword_dict.items():
        dist = levenshtein_distance(seg_str, ref_str)
        if dist < min_distance:
            min_distance = dist
            best_keyword = keyword

    if min_distance > distance_threshold:
        best_keyword = "no_keyword"

    return best_keyword, int(min_distance)


