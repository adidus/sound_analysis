import argparse
import json
import os
from typing import Dict, List

from pydub import AudioSegment

from .classifier import load_keyword_dictionary, recognize_keyword
from .evaluate import evaluate_recognition
from .features import fingerprint_from_file
from .segmentation import save_audio_segments, split_audio_by_silence


def cmd_build_dict(args: argparse.Namespace) -> None:
    d = load_keyword_dictionary(args.reference_dir)
    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"Saved dictionary with {len(d)} entries to {out_path}")


def cmd_split(args: argparse.Namespace) -> None:
    segments = split_audio_by_silence(
        args.audio,
        min_silence_len=args.min_silence_ms,
        silence_thresh=args.silence_thresh_dbfs,
        keep_silence=args.keep_silence_ms,
    )
    save_audio_segments(args.audio, segments)


def _load_dict_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cmd_recognize(args: argparse.Namespace) -> None:
    d = _load_dict_json(args.dictionary)
    segments = split_audio_by_silence(
        args.audio,
        min_silence_len=args.min_silence_ms,
        silence_thresh=args.silence_thresh_dbfs,
        keep_silence=args.keep_silence_ms,
    )
    for i, seg in enumerate(segments):
        keyword, dist = recognize_keyword(seg, d, distance_threshold=args.threshold)
        print(f"Segment {i}: {keyword} (dist={dist})")


def cmd_evaluate(args: argparse.Namespace) -> None:
    d = _load_dict_json(args.dictionary)
    with open(args.labels, "r", encoding="utf-8") as f:
        labels: List[str] = [line.strip() for line in f if line.strip()]
    evaluate_recognition(args.audio, labels, d)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kws",
        description=(
            "Keyword spotting tools: segmentation, fingerprinting, and deterministic classification"
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_dict = sub.add_parser("build-dict", help="Build dictionary from reference audio directory")
    p_dict.add_argument("reference_dir", help="Directory with keyword reference audio files")
    p_dict.add_argument("--output", default="dictionary.json", help="Output JSON path")
    p_dict.set_defaults(func=cmd_build_dict)

    p_split = sub.add_parser("split", help="Split an audio file by silence")
    p_split.add_argument("audio", help="Input audio file path")
    p_split.add_argument("--min-silence-ms", type=int, default=500)
    p_split.add_argument("--silence-thresh-dbfs", type=int, default=-35)
    p_split.add_argument("--keep-silence-ms", type=int, default=100)
    p_split.set_defaults(func=cmd_split)

    p_rec = sub.add_parser("recognize", help="Recognize keywords in an audio file")
    p_rec.add_argument("audio", help="Input audio file path")
    p_rec.add_argument("--dictionary", default="dictionary.json")
    p_rec.add_argument("--threshold", type=int, default=3000)
    p_rec.add_argument("--min-silence-ms", type=int, default=500)
    p_rec.add_argument("--silence-thresh-dbfs", type=int, default=-35)
    p_rec.add_argument("--keep-silence-ms", type=int, default=100)
    p_rec.set_defaults(func=cmd_recognize)

    p_eval = sub.add_parser("evaluate", help="Evaluate predictions vs labels file")
    p_eval.add_argument("audio", help="Input audio file path")
    p_eval.add_argument("--dictionary", default="dictionary.json")
    p_eval.add_argument("--labels", required=True, help="Text file: one expected keyword per line")
    p_eval.set_defaults(func=cmd_evaluate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


