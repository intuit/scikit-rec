"""skrec: A scikit-style recommender systems library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scikit-rec")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
