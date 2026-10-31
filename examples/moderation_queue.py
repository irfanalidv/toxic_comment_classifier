"""Moderation queue: auto-approve, hold for review, or reject incoming comments.

Shows: a small, explicit moderation policy built on ``classify_batch()``.
  - any label >= REJECT_AT              -> reject
  - ``threat`` or ``identity_hate`` >= ESCALATE_AT -> escalate to a human first
  - any label >= REVIEW_AT              -> hold for review
  - otherwise                           -> approve
Needs: only the package (offline). The thresholds are illustrative; tune them on your own
labelled data before using them for real decisions.

Run: python examples/moderation_queue.py
"""

from __future__ import annotations

from toxic_classifier import ToxicCommentClassifier

REJECT_AT = 0.9
REVIEW_AT = 0.5
ESCALATE_AT = 0.3


def decide(scores: dict[str, float]) -> tuple[str, list[str]]:
    """Return (decision, reasons) for one comment's label scores."""
    if any(scores[k] >= ESCALATE_AT for k in ("threat", "identity_hate")):
        return "escalate", [k for k in ("threat", "identity_hate") if scores[k] >= ESCALATE_AT]
    over_reject = [k for k, v in scores.items() if v >= REJECT_AT]
    if over_reject:
        return "reject", over_reject
    over_review = [k for k, v in scores.items() if v >= REVIEW_AT]
    if over_review:
        return "review", over_review
    return "approve", []


def main() -> None:
    incoming = [
        "Thanks, this fixed my issue!",
        "You are so dumb and stupid!",
        "Shut up and stop posting garbage.",
        "I disagree, but I see your point.",
        "You're disgusting!",
    ]
    clf = ToxicCommentClassifier()
    for text, scores in zip(incoming, clf.classify_batch(incoming), strict=True):
        decision, reasons = decide(scores)
        why = f" ({', '.join(reasons)})" if reasons else ""
        print(f"{decision:<9}{why:<28} {text}")


if __name__ == "__main__":
    main()
