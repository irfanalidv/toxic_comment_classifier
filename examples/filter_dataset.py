"""Dataset filtering: drop toxic rows from a CSV before using it for training.

Shows: scoring a whole CSV column with one batched call and writing clean / removed files.
Needs: only the package (offline). Uses the bundled ``examples/data/comments.csv``.

Run: python examples/filter_dataset.py [input.csv] [--max-toxicity 0.2]
Writes ``<input>.clean.csv`` and ``<input>.removed.csv`` to a new temporary directory
(nothing in the repository is modified).
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path

from toxic_classifier import ToxicCommentClassifier

DATA = Path(__file__).resolve().parent / "data" / "comments.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", type=Path, default=DATA)
    parser.add_argument("--column", default="text")
    parser.add_argument("--max-toxicity", type=float, default=0.2)
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scores = ToxicCommentClassifier().predict_batch([r[args.column] for r in rows])

    kept, removed = [], []
    for row, score in zip(rows, scores, strict=True):
        row["toxicity"] = f"{score:.4f}"
        (kept if score <= args.max_toxicity else removed).append(row)

    out_dir = Path(tempfile.mkdtemp(prefix="toxic-filter-"))
    for name, subset in (("clean", kept), ("removed", removed)):
        path = out_dir / f"{args.csv.stem}.{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*rows[0].keys()])
            writer.writeheader()
            writer.writerows(subset)
        print(f"{len(subset):>3} rows -> {path}")

    for row in removed:
        print(f"  removed  {row['toxicity']}  {row[args.column]}")


if __name__ == "__main__":
    main()
