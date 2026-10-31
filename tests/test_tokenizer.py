"""Tokenizer/padding tests (no TensorFlow needed)."""

import numpy as np
import pytest

from toxic_classifier.model import MAXLEN, MODEL_DIR, VOCAB_SIZE
from toxic_classifier.tokenizer import WordTokenizer, pad_sequences
from toxic_classifier.utils import clean_text


@pytest.fixture(scope="module")
def tok() -> WordTokenizer:
    return WordTokenizer.from_json(MODEL_DIR / "tokenizer.json")


def test_vocab_fits_embedding(tok: WordTokenizer) -> None:
    assert tok.num_words == VOCAB_SIZE
    assert max(tok.word_index.values()) < VOCAB_SIZE
    assert min(tok.word_index.values()) >= 1


def test_keras_style_splitting(tok: WordTokenizer) -> None:
    assert tok.text_to_words("Hello, WORLD!!  you're\tgreat") == [
        "hello",
        "world",
        "you're",
        "great",
    ]


def test_known_and_unknown_words(tok: WordTokenizer) -> None:
    seq = tok.text_to_sequence("the qwxzzyq stupid")
    assert len(seq) == 2  # unknown word dropped (no OOV token)
    assert seq[0] == tok.word_index["the"]


def test_oov_token_support() -> None:
    t = WordTokenizer({"<oov>": 1, "a": 2}, oov_token="<oov>")
    assert t.text_to_sequence("a b") == [2, 1]


def test_num_words_cutoff() -> None:
    t = WordTokenizer({"a": 1, "b": 5}, num_words=3)
    assert t.text_to_sequence("a b") == [1]


def test_pad_sequences_pre_padding_and_truncation() -> None:
    out = pad_sequences([[1, 2], [], list(range(1, 151))], maxlen=MAXLEN)
    assert out.shape == (3, MAXLEN) and out.dtype == np.int32
    assert out[0, -2:].tolist() == [1, 2] and out[0, :-2].sum() == 0
    assert out[1].sum() == 0
    assert out[2, 0] == 51 and out[2, -1] == 150  # keeps the last MAXLEN tokens


def test_clean_text() -> None:
    assert clean_text("Check http://x.com NOW!!  www.y.org ok") == "check now!! ok"
    assert clean_text("Wow—really?") == "wow really?"


def test_clean_text_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="expects a str"):
        clean_text(None)  # type: ignore[arg-type]


def test_char_level_tokenizer_rejected(tmp_path) -> None:
    path = tmp_path / "tok.json"
    path.write_text('{"config": {"char_level": true}, "word_index": {"a": 1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Character-level"):
        WordTokenizer.from_json(path)


def test_from_json_ignores_unknown_config_keys(tmp_path) -> None:
    path = tmp_path / "tok.json"
    path.write_text(
        '{"config": {"num_words": 10, "document_count": 3}, "word_index": {"hi": 1}}',
        encoding="utf-8",
    )
    tok = WordTokenizer.from_json(path)
    assert tok.text_to_sequence("Hi there") == [1]
