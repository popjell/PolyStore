"""
ROI conversion utilities for streaming backends and viewer servers.

Provides a single source of truth for converting ROI objects to:
- Napari shapes format
- ImageJ ROI bytes
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from .roi import EllipseShape, PointShape, PolygonShape, PolylineShape, ROI
from .streaming_constants import NapariShapeType

logger = logging.getLogger(__name__)


class NapariROIConverter:
    """Convert ROI objects to Napari shapes format."""

    _SHAPE_DIMENSION_HANDLERS = {
        "polygon": lambda shape_dict, prepend_dims: np.hstack(
            [np.tile(prepend_dims, (len(shape_dict["coordinates"]), 1)), np.array(shape_dict["coordinates"])]
        ),
        "polyline": lambda shape_dict, prepend_dims: np.hstack(
            [np.tile(prepend_dims, (len(shape_dict["coordinates"]), 1)), np.array(shape_dict["coordinates"])]
        ),
        "ellipse": lambda shape_dict, prepend_dims: np.hstack(
            [
                np.tile(prepend_dims, (4, 1)),
                np.array(
                    [
                        [
                            shape_dict["center"][0] - shape_dict["radii"][0],
                            shape_dict["center"][1] - shape_dict["radii"][1],
                        ],
                        [
                            shape_dict["center"][0] - shape_dict["radii"][0],
                            shape_dict["center"][1] + shape_dict["radii"][1],
                        ],
                        [
                            shape_dict["center"][0] + shape_dict["radii"][0],
                            shape_dict["center"][1] + shape_dict["radii"][1],
                        ],
                        [
                            shape_dict["center"][0] + shape_dict["radii"][0],
                            shape_dict["center"][1] - shape_dict["radii"][1],
                        ],
                    ]
                ),
            ]
        ),
        "point": lambda shape_dict, prepend_dims: np.concatenate([prepend_dims, shape_dict["coordinates"]]).reshape(1, -1),
    }

    @staticmethod
    def add_dimensions_to_shape(shape_dict: Dict[str, Any], prepend_dims: List[float]) -> np.ndarray:
        """Add dimensions to a 2D shape to make it nD."""
        shape_type = shape_dict["type"]
        shape_type_enum = NapariShapeType(shape_type) if isinstance(shape_type, str) else shape_type
        handler = NapariROIConverter._SHAPE_DIMENSION_HANDLERS.get(shape_type_enum.value)
        if handler is None:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        return handler(shape_dict, np.array(prepend_dims))

    @staticmethod
    def rois_to_shapes(rois: List[ROI]) -> List[Dict[str, Any]]:
        """Convert ROI objects to Napari shapes data."""
        shapes_data = []
        for roi in rois:
            if roi.shapes and all(isinstance(shape, PointShape) for shape in roi.shapes):
                points = [[shape.y, shape.x] for shape in roi.shapes]
                shapes_data.append({"type": "points", "coordinates": points, "metadata": roi.metadata})
            else:
                for shape in roi.shapes:
                    if isinstance(shape, PolygonShape):
                        shapes_data.append(
                            {"type": "polygon", "coordinates": shape.coordinates.tolist(), "metadata": roi.metadata}
                        )
                    elif isinstance(shape, PolylineShape):
                        shapes_data.append(
                            {"type": "path", "coordinates": shape.coordinates.tolist(), "metadata": roi.metadata}
                        )
                    elif isinstance(shape, EllipseShape):
                        shapes_data.append(
                            {
                                "type": "ellipse",
                                "center": [shape.center_y, shape.center_x],
                                "radii": [shape.radius_y, shape.radius_x],
                                "metadata": roi.metadata,
                            }
                        )
                    elif isinstance(shape, PointShape):
                        shapes_data.append({"type": "point", "coordinates": [shape.y, shape.x], "metadata": roi.metadata})
        return shapes_data

    @staticmethod
    def shapes_to_napari_format(shapes_data: List[Dict]) -> Tuple[List[np.ndarray], List[str], Dict]:
        """Convert shape dicts to Napari layer format."""
        napari_shapes = []
        shape_types = []
        properties = {"label": [], "area": [], "centroid_y": [], "centroid_x": []}

        for shape_dict in shapes_data:
            shape_type = shape_dict.get("type")
            metadata = shape_dict.get("metadata", {})

            if shape_type == "polygon":
                coords = np.array(shape_dict["coordinates"])
                napari_shapes.append(coords)
                shape_types.append("polygon")
                centroid = metadata.get("centroid", (0, 0))
                properties["label"].append(metadata.get("label", ""))
                properties["area"].append(metadata.get("area", 0))
                properties["centroid_y"].append(centroid[0])
                properties["centroid_x"].append(centroid[1])

            elif shape_type == "ellipse":
                center = np.array(shape_dict["center"])
                radii = np.array(shape_dict["radii"])
                corners = np.array([center - radii, center + radii])
                napari_shapes.append(corners)
                shape_types.append("ellipse")
                centroid = metadata.get("centroid", (0, 0))
                properties["label"].append(metadata.get("label", ""))
                properties["area"].append(metadata.get("area", 0))
                properties["centroid_y"].append(centroid[0])
                properties["centroid_x"].append(centroid[1])

            elif shape_type == "point":
                coords = np.array([shape_dict["coordinates"]])
                napari_shapes.append(coords)
                shape_types.append("point")
                point_coords = shape_dict["coordinates"]
                properties["label"].append(metadata.get("label", ""))
                properties["area"].append(0)
                properties["centroid_y"].append(point_coords[0])
                properties["centroid_x"].append(point_coords[1])

        return napari_shapes, shape_types, properties


class FijiROIConverter:
    """Convert ROI objects to ImageJ ROI bytes."""

    @staticmethod
    def rois_to_imagej_bytes(rois: List[ROI], roi_prefix: str = "") -> List[bytes]:
        """Convert ROI objects to ImageJ ROI bytes."""
        try:
            from roifile import ImagejRoi, ROI_TYPE
        except ImportError:
            raise ImportError("roifile library required for ImageJ ROI conversion. Install with: pip install roifile")

        roi_bytes_list = []
        for roi in rois:
            for shape in roi.shapes:
                if isinstance(shape, PolygonShape):
                    coords_xy = shape.coordinates[:, [1, 0]]
                    ij_roi = ImagejRoi.frompoints(coords_xy)
                    ij_roi.name = f"{roi_prefix}_ROI_{roi.metadata.get('label', '')}".rstrip("_")
                    roi_bytes_list.append(ij_roi.tobytes())
                elif isinstance(shape, PolylineShape):
                    coords_xy = shape.coordinates[:, [1, 0]]
                    ij_roi = ImagejRoi.frompoints(coords_xy)
                    ij_roi.roitype = ROI_TYPE.POLYLINE
                    ij_roi.name = f"{roi_prefix}_ROI_{roi.metadata.get('label', '')}".rstrip("_")
                    roi_bytes_list.append(ij_roi.tobytes())
                elif isinstance(shape, EllipseShape):
                    center_x = shape.center_x
                    center_y = shape.center_y
                    radius_x = shape.radius_x
                    radius_y = shape.radius_y
                    left = center_x - radius_x
                    top = center_y - radius_y
                    width = 2 * radius_x
                    height = 2 * radius_y
                    ij_roi = ImagejRoi.frompoints(np.array([[left, top], [left + width, top + height]]))
                    ij_roi.roitype = ImagejRoi.OVAL if hasattr(ImagejRoi, "OVAL") else ROI_TYPE.OVAL
                    ij_roi.name = f"{roi_prefix}_ROI_{roi.metadata.get('label', '')}".rstrip("_")
                    roi_bytes_list.append(ij_roi.tobytes())
                elif isinstance(shape, PointShape):
                    coords_xy = np.array([[shape.x, shape.y]])
                    ij_roi = ImagejRoi.frompoints(coords_xy)
                    ij_roi.name = f"{roi_prefix}_ROI_{roi.metadata.get('label', '')}".rstrip("_")
                    roi_bytes_list.append(ij_roi.tobytes())

        return roi_bytes_list

    @staticmethod
    def encode_rois_for_transmission(roi_bytes_list: List[bytes]) -> List[str]:
        """Base64 encode ROI bytes for JSON transmission."""
        import base64
        return [base64.b64encode(roi_bytes).decode("utf-8") for roi_bytes in roi_bytes_list]

    @staticmethod
    def decode_rois_from_transmission(encoded_rois: List[str]) -> List[bytes]:
        """Decode base64-encoded ROI bytes."""
        import base64
        return [base64.b64decode(roi_encoded) for roi_encoded in encoded_rois]

    @staticmethod
    def bytes_to_java_roi(roi_bytes: bytes, scyjava_module) -> Any:
        """Convert ROI bytes to Java ROI object via temporary file."""
        import os
        import tempfile

        RoiDecoder = scyjava_module.jimport("ij.io.RoiDecoder")
        with tempfile.NamedTemporaryFile(suffix=".roi", delete=False) as tmp:
            tmp.write(roi_bytes)
            tmp_path = tmp.name

        try:
            roi_decoder = RoiDecoder(tmp_path)
            return roi_decoder.getRoi()
        finally:
            os.unlink(tmp_path)
