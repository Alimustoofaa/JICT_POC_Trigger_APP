from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from src.camera.axis import AxisCamera
    from src.camera.base import CameraError
    from src.camera.hikvision import HikvisionCamera
except ImportError:
    from camera.axis import AxisCamera
    from camera.base import CameraError
    from camera.hikvision import HikvisionCamera

POLYGON_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lane.json"
CAMERA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "cameras.json"


class PolygonConfigError(Exception):
    """Raised when the polygon config cannot be loaded or parsed."""


def load_camera_config(
    config_path: str | Path = CAMERA_CONFIG_PATH,
) -> dict[str, Any]:
    """Load camera definitions from the JSON config file."""
    try:
        with Path(config_path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        raise CameraError(
            "Failed to load camera config",
            {"config_path": str(config_path), "cause": str(error)},
        ) from error


def initialize_camera(
    camera_id: str,
    config_path: str | Path = CAMERA_CONFIG_PATH,
) -> HikvisionCamera | AxisCamera:
    """Create a camera instance from the configured camera id."""
    cameras = load_camera_config(config_path)
    config = cameras.get(camera_id)

    if not config:
        raise CameraError("Camera config not found", {"id": camera_id})

    brand = config.get("brand")
    if brand == "HIKVISION":
        return HikvisionCamera(camera_id, config)
    if brand == "AXIS":
        return AxisCamera(camera_id, config)

    raise CameraError(
        "Unsupported camera brand",
        {"id": camera_id, "brand": brand},
    )


def initialize_cameras(
    config_path: str | Path = CAMERA_CONFIG_PATH,
) -> dict[str, HikvisionCamera | AxisCamera]:
    """Create camera instances for all configured cameras."""
    cameras = load_camera_config(config_path)
    return {
        camera_id: initialize_camera(camera_id, config_path)
        for camera_id in cameras
    }


def initializetion_camera(
    config_path: str | Path = CAMERA_CONFIG_PATH,
) -> dict[str, HikvisionCamera | AxisCamera]:
    """Backward-friendly alias that returns initialized cameras as a dict."""
    return initialize_cameras(config_path)


def load_polygon_config(
    config_path: str | Path = POLYGON_CONFIG_PATH,
) -> dict[str, list[list[float]]]:
    """Load polygon points from a LabelMe-style JSON config."""
    try:
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception as error:
        raise PolygonConfigError(
            f"Failed to load polygon config from '{config_path}': {error}"
        ) from error

    shapes = config.get("shapes")
    if not isinstance(shapes, list):
        raise PolygonConfigError("Invalid polygon config: 'shapes' must be a list")

    polygons: dict[str, list[list[float]]] = {}

    for shape in shapes:
        if not isinstance(shape, dict):
            continue

        if shape.get("shape_type") != "polygon":
            continue

        label = shape.get("label")
        points = shape.get("points")

        if not label or not isinstance(points, list):
            continue

        polygons[label] = points

    if not polygons:
        raise PolygonConfigError("No polygon shapes found in config")

    return polygons


__all__ = [
    "CAMERA_CONFIG_PATH",
    "POLYGON_CONFIG_PATH",
    "PolygonConfigError",
    "load_camera_config",
    "initialize_camera",
    "initialize_cameras",
    "initializetion_camera",
    "load_polygon_config",
]
