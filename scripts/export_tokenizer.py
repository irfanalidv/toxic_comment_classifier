"""Convert the original pickled Keras ``Tokenizer`` into the JSON vocabulary shipped with
the package (``toxic_classifier/model/tokenizer.json``).

Only words the model can actually see are kept (index < ``num_words``). Requires Keras 3,
because the pickle references ``keras.src.legacy.preprocessing.text.Tokenizer``.

Usage:
    python scripts/export_tokenizer.py tokenizer.pickle toxic_classifier/model/tokenizer.json

(The original ``tokenizer.pickle`` is in the git history, release 0.3.0.)
"""

from __future__ import annotations

import json
import pickle
import sys


def main(src: str, dst: str) -> None:
    with open(src, "rb") as handle:
        tok = pickle.load(handle)  # noqa: S301 - trusted, project-owned artifact

    num_words = tok.num_words
    word_index = {
        word: idx for word, idx in tok.word_index.items() if num_words is None or idx < num_words
    }
    payload = {
        "config": {
            "num_words": num_words,
            "filters": tok.filters,
            "lower": tok.lower,
            "split": tok.split,
            "char_level": tok.char_level,
            "oov_token": tok.oov_token,
        },
        "word_index": dict(sorted(word_index.items(), key=lambda kv: kv[1])),
    }
    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {len(word_index)} words to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
