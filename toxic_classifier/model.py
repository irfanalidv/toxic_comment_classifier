"""Pre-trained BiLSTM toxic comment classifier (Jigsaw labels)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .tokenizer import WordTokenizer, pad_sequences
from .utils import clean_text

# === Constants (must match training) ===
MAXLEN = 100
VOCAB_SIZE = 20000
EMBEDDING_DIM = 50
LIST_CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MODEL_DIR = Path(__file__).resolve().parent / "model"


def _build_model():
    """Rebuild the training architecture (Keras 3)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silence TensorFlow C++ info logs
    import keras
    from keras import layers

    inp = keras.Input(shape=(MAXLEN,))
    x = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, trainable=False)(inp)
    x = layers.Bidirectional(
        layers.LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    )(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(50, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(len(LIST_CLASSES), activation="sigmoid")(x)
    return keras.Model(inputs=inp, outputs=output)


class ToxicCommentClassifier:
    """Multi-label toxicity classifier.

    Scores six Jigsaw toxicity labels (``toxic``, ``severe_toxic``, ``obscene``,
    ``threat``, ``insult``, ``identity_hate``) with values between 0 and 1.

    Args:
        clean: If True, apply :func:`toxic_classifier.utils.clean_text` (lowercase,
            strip URLs and special characters) before scoring. Default False, which
            matches the behaviour of earlier releases.
        batch_size: Batch size used for model inference on lists of texts.
    """

    labels: tuple[str, ...] = tuple(LIST_CLASSES)

    def __init__(self, clean: bool = False, batch_size: int = 256) -> None:
        self.clean = clean
        self.batch_size = batch_size
        self.tokenizer = WordTokenizer.from_json(MODEL_DIR / "tokenizer.json")
        self.model = _build_model()
        self.model.load_weights(str(MODEL_DIR / "model.weights.h5"))

    # ----------------------------------------------------------------- helpers
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        prepared = [clean_text(t) if self.clean else t for t in texts]
        return pad_sequences(self.tokenizer.texts_to_sequences(prepared), maxlen=MAXLEN)

    def _scores(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("Expected a list of strings; use classify() for a single text")
        if len(texts) == 0:
            return np.zeros((0, len(LIST_CLASSES)), dtype="float32")
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"All inputs must be strings, got {type(t).__name__}")
        return np.asarray(
            self.model.predict(self._encode(texts), batch_size=self.batch_size, verbose=0)
        )

    # --------------------------------------------------------------------- API
    def classify(self, text: str) -> dict[str, float]:
        """Return per-label toxicity scores (0-1) for a single text."""
        return self.classify_batch([text])[0]

    def classify_batch(self, texts: Sequence[str]) -> list[dict[str, float]]:
        """Return per-label toxicity scores for each text (one model call per batch)."""
        return [
            {label: float(score) for label, score in zip(LIST_CLASSES, row, strict=True)}
            for row in self._scores(texts)
        ]

    def predict(self, text: str) -> float:
        """Return the overall toxicity score: the mean of the six label scores."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[float]:
        """Return the overall toxicity score (mean of label scores) for each text."""
        return [float(row.mean()) for row in self._scores(texts)]

    def flagged_labels(self, text: str, threshold: float = 0.5) -> list[str]:
        """Return the labels whose score is at or above ``threshold``."""
        return [label for label, score in self.classify(text).items() if score >= threshold]
