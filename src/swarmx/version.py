"""Runtime version information."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("swarmx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
