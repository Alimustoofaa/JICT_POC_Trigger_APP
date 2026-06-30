from __future__ import annotations

from base64 import b64encode
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlencode, urljoin
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
)


class CameraError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass
class BaseCamera:
    camera_id: str
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.camera_id:
            raise CameraError("Camera id is required")

        if not isinstance(self.config, dict):
            raise CameraError("Camera config is required", {"id": self.camera_id})

        required_fields = (
            "ip",
            "username",
            "password",
            "stream_endpoint",
            "capture_endpoint",
            "brand",
        )

        for field in required_fields:
            if not self.config.get(field):
                raise CameraError(
                    f"Missing camera config field: {field}",
                    {"id": self.camera_id, "field": field},
                )

    @property
    def name(self) -> str:
        return self.config.get("name", self.camera_id)

    @property
    def brand(self) -> str:
        return self.config["brand"]

    @property
    def zone(self) -> str | None:
        return self.config.get("zone")

    @property
    def host(self) -> str:
        return self.config["ip"]

    @property
    def protocol(self) -> str:
        return self.config.get("protocol", "http")

    def get_auth_header(self) -> str:
        token = b64encode(
            f"{self.config['username']}:{self.config['password']}".encode("utf-8")
        ).decode("ascii")
        return f"Basic {token}"

    def build_url(self, pathname: str, query: dict[str, Any] | None = None) -> str:
        base = f"{self.protocol}://{self.host}"
        url = urljoin(f"{base}/", pathname.lstrip("/"))

        if not query:
            return url

        filtered_query = {
            key: str(value)
            for key, value in query.items()
            if value is not None
        }
        return f"{url}?{urlencode(filtered_query)}"

    def get_stream_url(self, query: dict[str, Any] | None = None) -> str:
        return self.build_url(self.config["stream_endpoint"], query)

    def get_rtsp_url(self, query: dict[str, Any] | None = None) -> str:
        credentials = (
            f"{quote(self.config['username'])}:{quote(self.config['password'])}@"
        )
        base = f"rtsp://{credentials}{self.host}"
        url = urljoin(f"{base}/", self.config["stream_endpoint"].lstrip("/"))

        if not query:
            return url

        filtered_query = {
            key: str(value)
            for key, value in query.items()
            if value is not None
        }
        return f"{url}?{urlencode(filtered_query)}"

    def get_capture_url(self, query: dict[str, Any] | None = None) -> str:
        return self.build_url(self.config["capture_endpoint"], query)

    def request(
        self,
        pathname: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> bytes:
        url = self.build_url(pathname)
        password_manager = HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(
            realm=None,
            uri=url,
            user=self.config["username"],
            passwd=self.config["password"],
        )

        opener = build_opener(
            HTTPDigestAuthHandler(password_manager),
            HTTPBasicAuthHandler(password_manager),
        )

        request = Request(
            url,
            data=body,
            headers=headers or {},
            method=method,
        )

        try:
            with opener.open(request, timeout=10) as response:
                return response.read()
        except Exception as error:
            raise CameraError(
                "Camera request failed",
                {
                    "id": self.camera_id,
                    "pathname": pathname,
                    "method": method,
                    "cause": str(error),
                },
            ) from error

    def get_snapshot_buffer(self) -> bytes:
        return self.request(self.config["capture_endpoint"], method="GET")

    def get_capabilities(self) -> dict[str, bool]:
        return {
            "stream": True,
            "snapshot": True,
        }

    def apply_default_view(self) -> None:
        """Apply optional default camera positioning from config."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.camera_id,
            "name": self.name,
            "brand": self.brand,
            "zone": self.zone,
            "host": self.host,
            "capabilities": self.get_capabilities(),
            "stream_url": self.get_stream_url(),
            "rtsp_url": self.get_rtsp_url(),
            "capture_url": self.get_capture_url(),
        }
