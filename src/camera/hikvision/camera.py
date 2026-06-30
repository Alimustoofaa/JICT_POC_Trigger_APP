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

    def apply_default_view(self) -> None:
        if not self.config.get("ptz_enabled", False):
            return

        zoom_speed = int(self.config.get("default_zoom_speed", 0))
        zoom_duration = float(self.config.get("default_zoom_duration", 0))
        pan_speed = int(self.config.get("default_pan_speed", 0))
        pan_duration = float(self.config.get("default_pan_duration", 0))
        tilt_speed = int(self.config.get("default_tilt_speed", 0))
        tilt_duration = float(self.config.get("default_tilt_duration", 0))

        has_default_ptz = any(
            (
                pan_speed,
                pan_duration,
                tilt_speed,
                tilt_duration,
                zoom_speed,
                zoom_duration,
            )
        )
        if not has_default_ptz:
            return

        # Stop any ongoing PTZ movement first only when PTZ defaults are configured.
        self.stop()

        if pan_speed:
            self.continuous_move(pan=pan_speed)
            from time import sleep
            sleep(max(0.0, pan_duration))
            self.stop()

        if tilt_speed:
            self.continuous_move(tilt=tilt_speed)
            from time import sleep
            sleep(max(0.0, tilt_duration))
            self.stop()

        if zoom_speed:
            self.continuous_move(zoom=zoom_speed)
            from time import sleep
            sleep(max(0.0, zoom_duration))
            self.stop()

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
