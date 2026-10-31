"""Pre-trained deep-learning classifier for toxic comments (Jigsaw labels)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from .model import LIST_CLASSES, ToxicCommentClassifier
from .utils import clean_text

try:
    __version__ = _pkg_version("toxic_comment_classifier")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["LIST_CLASSES", "ToxicCommentClassifier", "__version__", "clean_text"]
