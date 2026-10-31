"""Text normalisation helpers."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"http\S+|www\.\S+")
_DISALLOWED_RE = re.compile(r"[^a-zA-Z0-9!?.,]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase ``text`` and remove URLs, special characters and extra whitespace.

    Only ASCII letters, digits and ``! ? . ,`` are kept; everything else becomes a space.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text() expects a str, got {type(text).__name__}")
    text = _URL_RE.sub("", text.lower())
    text = _DISALLOWED_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
