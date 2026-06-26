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

CAMERA_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "cameras.json"


def load_camera_config(config_path: str | Path = CAMERA_CONFIG_PATH) -> dict[str, Any]:
    try:
        with Path(config_path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as error:
        raise CameraError(
            "Failed to load camera config",
            {"config_path": str(config_path), "cause": str(error)},
        ) from error


def create_camera(camera_id: str, config: dict[str, Any]) -> HikvisionCamera | AxisCamera:
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


def get_camera(camera_id: str, config_path: str | Path = CAMERA_CONFIG_PATH) -> HikvisionCamera | AxisCamera:
    cameras = load_camera_config(config_path)
    return create_camera(camera_id, cameras.get(camera_id))


def get_all_cameras(
    config_path: str | Path = CAMERA_CONFIG_PATH,
) -> dict[str, HikvisionCamera | AxisCamera]:
    cameras = load_camera_config(config_path)
    return {
        camera_id: create_camera(camera_id, config)
        for camera_id, config in cameras.items()
    }


__all__ = [
    "AxisCamera",
    "CameraError",
    "HikvisionCamera",
    "create_camera",
    "get_all_cameras",
    "get_camera",
    "load_camera_config",
]
