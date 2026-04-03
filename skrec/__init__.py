"""skrec: A scikit-style recommender systems library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scikit-rec")
except PackageNotFoundError:
    __version__ = "unknown"
