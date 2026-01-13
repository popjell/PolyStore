"""
Utility functions for polystore.
"""

import re
from pathlib import Path
from typing import List, Union


def natural_sort_key(text: Union[str, Path]) -> List[Union[str, int]]:
    """Generate a natural sorting key for a string or Path."""
    text = str(text)
    parts = re.split(r"(\d+)", text)
    result = []
    for part in parts:
        result.append(int(part) if part.isdigit() else part)
    return result


def natural_sort(items: List[Union[str, Path]]) -> List[Union[str, Path]]:
    """Sort strings or Paths using natural ordering."""
    return sorted(items, key=natural_sort_key)


def natural_sort_inplace(items: List[Union[str, Path]]) -> None:
    """Sort a list of strings or Paths in-place using natural ordering."""
    items.sort(key=natural_sort_key)


def get_zmq_transport_url(port: int, transport_mode: str = "tcp") -> str:
    """Get ZeroMQ transport URL (simple fallback utility)."""
    if transport_mode == "tcp":
        return f"tcp://127.0.0.1:{port}"
    if transport_mode == "ipc":
        return f"ipc:///tmp/polystore_{port}.ipc"
    raise ValueError(f"Unknown transport mode: {transport_mode}")
