"""End-to-end tests with the bundled pre-trained model (requires TensorFlow/Keras)."""

import json

import pytest

import toxic_classifier
from toxic_classifier import LIST_CLASSES, ToxicCommentClassifier
from toxic_classifier.cli import main


@pytest.fixture(scope="module")
def clf() -> ToxicCommentClassifier:
    return ToxicCommentClassifier()


def test_version() -> None:
    from importlib.metadata import version

    assert toxic_classifier.__version__ == version("toxic_comment_classifier")


def test_classify_returns_all_labels(clf: ToxicCommentClassifier) -> None:
    scores = clf.classify("You are so dumb and stupid!")
    assert list(scores) == LIST_CLASSES
    assert all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in scores.values())


def test_reference_scores_match_published_example(clf: ToxicCommentClassifier) -> None:
    # Values documented in the README (produced by the original 0.3.0 implementation)
    scores = clf.classify("You are so dumb and stupid!")
    assert scores["toxic"] == pytest.approx(0.98894, abs=1e-4)
    assert scores["insult"] == pytest.approx(0.86641, abs=1e-4)
    assert clf.predict("You are so dumb and stupid!") == pytest.approx(0.4347, abs=1e-4)


def test_toxic_scores_higher_than_benign(clf: ToxicCommentClassifier) -> None:
    toxic, benign = clf.predict_batch(["You are so dumb and stupid!", "Have a nice day :)"])
    assert toxic > 0.3 > 0.05 > benign


def test_batch_matches_single(clf: ToxicCommentClassifier) -> None:
    texts = ["I hate you so much!", "This is wonderful news.", ""]
    batch = clf.classify_batch(texts)
    for text, scores in zip(texts, batch, strict=True):
        single = clf.classify(text)
        assert single == pytest.approx(scores, abs=1e-5)
    assert clf.predict_batch([]) == []


def test_predict_returns_python_float(clf: ToxicCommentClassifier) -> None:
    assert type(clf.predict("hello")) is float


def test_input_validation(clf: ToxicCommentClassifier) -> None:
    with pytest.raises(TypeError):
        clf.predict_batch("not a list")
    with pytest.raises(TypeError):
        clf.classify_batch(["ok", None])


def test_flagged_labels(clf: ToxicCommentClassifier) -> None:
    assert "toxic" in clf.flagged_labels("You are so dumb and stupid!")
    assert clf.flagged_labels("Have a nice day :)") == []


def test_cli_outputs_json_lines(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["You are so dumb and stupid!", "Have a nice day"]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    assert len(lines) == 2
    assert "toxic" in lines[0]["flagged"]
    assert set(lines[1]["scores"]) == set(LIST_CLASSES)
