"""Validated IRAC channel metadata shared by ingestion and modeling."""
from __future__ import annotations

import os
import re

from . import config


def channel_from_filename(filename: str) -> str | None:
    """Return a channel from an IRAC filename token, or None if unspecified.

    Conflicting channel tokens are rejected rather than selecting a calibration
    arbitrarily. Directory names are intentionally not used as band metadata.
    """
    tokens = re.findall(r"(?:^|[_\-.])(ch[12]|i[12])(?=[_\-.]|$)",
                        os.path.basename(filename).lower())
    channels = {"ch" + token[-1] for token in tokens}
    if len(channels) > 1:
        raise ValueError(f"Conflicting IRAC channels in filename: {filename}")
    return next(iter(channels), None)


def frame_channel(entry: dict) -> str:
    """Resolve explicit metadata and filename; use configured band if neither exists."""
    inferred = channel_from_filename(str(entry.get("filename", "")))
    explicit = entry.get("channel")
    channel = str(explicit if explicit is not None else inferred or config.CHANNEL).lower()
    if channel in ("1", "2"):
        channel = "ch" + channel
    if channel not in ("ch1", "ch2"):
        raise ValueError(f"Unsupported IRAC channel: {channel!r}")
    if inferred is not None and channel != inferred:
        raise ValueError(f"Channel metadata {channel} disagrees with filename {entry.get('filename')}")
    return channel
