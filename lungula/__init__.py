"""lun'gula — game imitation learning framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lungula")
except PackageNotFoundError:
    __version__ = "unknown"
