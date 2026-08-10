"""Errors that cross only the worker-to-RSI Python call boundary."""


class RsiError(RuntimeError):
    """The private RSI MCP process failed or violated its contract."""


class RsiCancelledError(RsiError):
    """The worker cancelled an in-flight private RSI MCP call."""
