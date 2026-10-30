"""Failures shared by model adapters and the transport."""


class TooLong(Exception):
    """More labels than the model's context holds."""
