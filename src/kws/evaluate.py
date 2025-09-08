from typing import Dict, List

from pydub import AudioSegment
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from .classifier import recognize_keyword
from .segmentation import split_audio_by_silence


def evaluate_recognition(test_audio_file: str, expected_keywords: List[str], keyword_dict: Dict[str, str]) -> None:
    print(f"\nEvaluating {test_audio_file}...")

    segments = split_audio_by_silence(test_audio_file)
    predictions: List[str] = []

    for i, segment in enumerate(segments):
        predicted, dist = recognize_keyword(segment, keyword_dict)
        predictions.append(predicted)
        expected = expected_keywords[i] if i < len(expected_keywords) else "N/A"
        print(f"Segment {i}: predicted = {predicted} (dist={dist}), expected = {expected}")

    min_len = min(len(expected_keywords), len(predictions))
    y_true = expected_keywords[:min_len]
    y_pred = predictions[:min_len]

    print("\nConfusion Matrix:")
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for label, row in zip(labels, cm):
        print(f"{label:>10}: {row}")

    print("\nMetrics:")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.2f}")


