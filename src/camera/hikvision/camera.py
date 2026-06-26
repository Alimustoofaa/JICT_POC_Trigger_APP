from __future__ import annotations

try:
    from src.camera.base import BaseCamera, CameraError
except ImportError:
    from camera.base import BaseCamera, CameraError

HIKVISION_BRAND = "HIKVISION"
PTZ_ENDPOINT = "/ISAPI/PTZCtrl/channels/1/continuous"


class HikvisionCamera(BaseCamera):
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.config["brand"] != HIKVISION_BRAND:
            raise CameraError(
                "Invalid brand for Hikvision camera",
                {"id": self.camera_id, "brand": self.config["brand"]},
            )

    def get_capabilities(self) -> dict[str, bool]:
        return {
            **super().get_capabilities(),
            "pan": True,
            "tilt": True,
            "zoom": True,
            "ptz": True,
        }

    def continuous_move(self, pan: int = 0, tilt: int = 0, zoom: int = 0) -> None:
        self._assert_move_range("pan", pan)
        self._assert_move_range("tilt", tilt)
        self._assert_move_range("zoom", zoom)

        body = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<PTZData>"
            f"<pan>{pan}</pan>"
            f"<tilt>{tilt}</tilt>"
            f"<zoom>{zoom}</zoom>"
            "</PTZData>"
        ).encode("utf-8")

        self.request(
            PTZ_ENDPOINT,
            method="PUT",
            headers={"Content-Type": "application/xml"},
            body=body,
        )

    def stop(self) -> None:
        self.continuous_move(0, 0, 0)

    def pan_left(self, speed: int = 30) -> None:
        self.continuous_move(pan=-abs(speed))

    def pan_right(self, speed: int = 30) -> None:
        self.continuous_move(pan=abs(speed))

    def tilt_up(self, speed: int = 30) -> None:
        self.continuous_move(tilt=abs(speed))

    def tilt_down(self, speed: int = 30) -> None:
        self.continuous_move(tilt=-abs(speed))

    def zoom_in(self, speed: int = 20) -> None:
        self.continuous_move(zoom=abs(speed))

    def zoom_out(self, speed: int = 20) -> None:
        self.continuous_move(zoom=-abs(speed))

    def _assert_move_range(self, axis: str, value: int) -> None:
        if not isinstance(value, (int, float)) or value < -100 or value > 100:
            raise CameraError(
                f"Invalid {axis} speed",
                {
                    "id": self.camera_id,
                    "axis": axis,
                    "value": value,
                    "min": -100,
                    "max": 100,
                },
            )
