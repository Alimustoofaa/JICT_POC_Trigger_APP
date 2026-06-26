from __future__ import annotations

try:
    from src.camera.base import BaseCamera, CameraError
except ImportError:
    from camera.base import BaseCamera, CameraError

AXIS_BRAND = "AXIS"
PTZ_ENDPOINT = "/axis-cgi/com/ptz.cgi"


class AxisCamera(BaseCamera):
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.config["brand"] != AXIS_BRAND:
            raise CameraError(
                "Invalid brand for Axis camera",
                {"id": self.camera_id, "brand": self.config["brand"]},
            )

    def get_capabilities(self) -> dict[str, bool]:
        return {
            **super().get_capabilities(),
            "zoom": True,
            "pan": False,
            "tilt": False,
            "ptz": False,
        }

    def zoom_in(self, step: int = 50) -> None:
        self._relative_zoom(abs(step))

    def zoom_out(self, step: int = 50) -> None:
        self._relative_zoom(-abs(step))

    def set_zoom(self, level: int) -> None:
        if not isinstance(level, (int, float)) or level < 1 or level > 9999:
            raise CameraError(
                "Invalid Axis zoom level",
                {
                    "id": self.camera_id,
                    "level": level,
                    "min": 1,
                    "max": 9999,
                },
            )

        self.request(f"{PTZ_ENDPOINT}?zoom={round(level)}", method="GET")

    def _relative_zoom(self, amount: int) -> None:
        if not isinstance(amount, (int, float)) or amount < -9999 or amount > 9999:
            raise CameraError(
                "Invalid Axis relative zoom amount",
                {
                    "id": self.camera_id,
                    "amount": amount,
                    "min": -9999,
                    "max": 9999,
                },
            )

        self.request(f"{PTZ_ENDPOINT}?rzoom={round(amount)}", method="GET")
