"""Dependency-free text tokenizer compatible with the Keras ``Tokenizer`` used in training.

The vocabulary ships as JSON (``model/tokenizer.json``) so loading it does not depend on
Keras-internal class paths (which change between Keras versions) or on unpickling.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


class WordTokenizer:
    """Reimplementation of ``keras.preprocessing.text.Tokenizer.texts_to_sequences``."""

    def __init__(
        self,
        word_index: dict[str, int],
        num_words: int | None = None,
        filters: str = DEFAULT_FILTERS,
        lower: bool = True,
        split: str = " ",
        oov_token: str | None = None,
        **_: Any,
    ) -> None:
        self.word_index = word_index
        self.num_words = num_words
        self.filters = filters
        self.lower = lower
        self.split = split
        self.oov_token = oov_token
        self._translate = str.maketrans({c: split for c in filters})
        self._oov_index = word_index.get(oov_token) if oov_token else None

    @classmethod
    def from_json(cls, path: str | Path) -> WordTokenizer:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        config = data.get("config", {})
        if config.get("char_level"):
            raise ValueError("Character-level tokenizers are not supported")
        return cls(word_index=data["word_index"], **config)

    def text_to_words(self, text: str) -> list[str]:
        """Same splitting rules as ``keras.preprocessing.text.text_to_word_sequence``."""
        if self.lower:
            text = text.lower()
        text = text.translate(self._translate)
        return [w for w in text.split(self.split) if w]

    def text_to_sequence(self, text: str) -> list[int]:
        seq: list[int] = []
        for word in self.text_to_words(text):
            idx = self.word_index.get(word)
            if idx is not None and (self.num_words is None or idx < self.num_words):
                seq.append(idx)
            elif self._oov_index is not None:
                seq.append(self._oov_index)
        return seq

    def texts_to_sequences(self, texts: Iterable[str]) -> list[list[int]]:
        return [self.text_to_sequence(t) for t in texts]


def pad_sequences(sequences: list[list[int]], maxlen: int, value: int = 0) -> np.ndarray:
    """Left-pad and left-truncate sequences (Keras ``pad_sequences`` defaults)."""
    out = np.full((len(sequences), maxlen), value, dtype="int32")
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        trunc = seq[-maxlen:]
        out[i, -len(trunc) :] = trunc
    return out
