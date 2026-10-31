"""Community health report: share of flagged messages per day, channel and author.

Shows: turning per-comment scores into simple aggregate metrics for a dashboard.
Needs: only the package (offline). Uses the bundled ``examples/data/comments.csv``.

Run: python examples/community_health.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from toxic_classifier import ToxicCommentClassifier

DATA = Path(__file__).resolve().parent / "data" / "comments.csv"
THRESHOLD = 0.5


def share_flagged(rows: list[dict[str, str]], flagged: list[bool], key: str) -> dict[str, str]:
    totals: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for row, is_flagged in zip(rows, flagged, strict=True):
        totals[row[key]] += 1
        hits[row[key]] += is_flagged
    return {k: f"{hits[k]}/{totals[k]} ({hits[k] / totals[k]:.0%})" for k in sorted(totals)}


def main() -> None:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clf = ToxicCommentClassifier()
    label_scores = clf.classify_batch([r["text"] for r in rows])
    flagged = [any(v >= THRESHOLD for v in s.values()) for s in label_scores]

    print(f"Flagged overall: {sum(flagged)}/{len(rows)} (any label >= {THRESHOLD})")
    for key in ("date", "channel", "author"):
        print(f"\nBy {key}:")
        for value, summary in share_flagged(rows, flagged, key).items():
            print(f"  {value:<12} {summary}")

    counts: dict[str, int] = defaultdict(int)
    for scores in label_scores:
        for label, value in scores.items():
            counts[label] += value >= THRESHOLD
    print("\nLabel counts:", dict(counts))


if __name__ == "__main__":
    main()
