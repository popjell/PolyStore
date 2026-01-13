"""
Streaming-related enums for polystore.

Provides type-safe enums for data types and shape types used in streaming
backends and viewer integrations.
"""

from enum import Enum


class StreamingDataType(Enum):
    """Types of data that can be streamed to viewers."""
    IMAGE = "image"
    SHAPES = "shapes"  # Napari shapes layer
    POINTS = "points"  # Napari points layer (e.g., skeleton tracings)
    ROIS = "rois"      # Fiji ROI payloads


class NapariShapeType(Enum):
    """Napari shape types for ROI visualization."""
    POLYGON = "polygon"
    ELLIPSE = "ellipse"
    POINT = "point"
    LINE = "line"
    PATH = "path"
    RECTANGLE = "rectangle"
