"""Command-line interface: ``toxic-classifier "some text"``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="toxic-classifier",
        description="Score comments for toxicity (toxic, severe_toxic, obscene, threat, "
        "insult, identity_hate). Reads one comment per line from stdin if no text is given.",
    )
    parser.add_argument("texts", nargs="*", help="Comment(s) to classify")
    parser.add_argument("--clean", action="store_true", help="Normalize text before scoring")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score at which a label is reported as flagged (default: 0.5)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    texts = args.texts or [line.rstrip("\n") for line in sys.stdin if line.strip()]
    if not texts:
        parser.error("no input text given")

    from .model import ToxicCommentClassifier

    clf = ToxicCommentClassifier(clean=args.clean)
    for text, scores in zip(texts, clf.classify_batch(texts), strict=True):
        record = {
            "text": text,
            "toxicity": round(sum(scores.values()) / len(scores), 6),
            "flagged": [k for k, v in scores.items() if v >= args.threshold],
            "scores": {k: round(v, 6) for k, v in scores.items()},
        }
        print(json.dumps(record, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
