"""lun'imago — game imitation learning framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lunimago")
except PackageNotFoundError:
    __version__ = "unknown"
