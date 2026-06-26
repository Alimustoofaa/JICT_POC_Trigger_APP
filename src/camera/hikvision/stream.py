from __future__ import annotations

import select
import subprocess as sp
import sys
import time
from typing import Any

import numpy as np


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 15
DEFAULT_READ_TIMEOUT = 10


class FFmpegRTSPCamera:
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str | None = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        read_timeout: int = DEFAULT_READ_TIMEOUT,
    ) -> None:
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.read_timeout = read_timeout

        self.pipe: sp.Popen[bytes] | None = None
        self.frame_bytes = self.width * self.height * 3
        self.last_frame_time: float | None = None
        self.frame_count = 0

        self.command = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.rtsp_url,
            "-an",
            "-r",
            str(self.fps),
            "-s",
            f"{self.width}x{self.height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-loglevel",
            "quiet",
            "-",
        ]

    def open(self) -> None:
        print(f"[INFO] Opening RTSP stream: {self.camera_id or self.rtsp_url}")
        self.pipe = sp.Popen(
            self.command,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            bufsize=10**8,
        )
        self.last_frame_time = time.time()
        self.frame_count = 0

    def read(self) -> np.ndarray | None:
        if self.pipe is None or self.pipe.stdout is None:
            return None

        raw = self._read_with_timeout(
            self.pipe.stdout,
            self.frame_bytes,
            self.read_timeout,
        )

        if raw is None or len(raw) != self.frame_bytes:
            return None

        frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
        self.last_frame_time = time.time()
        self.frame_count += 1
        return frame

    def _read_with_timeout(
        self,
        pipe_stdout: Any,
        size: int,
        timeout: int,
    ) -> bytes | None:
        if sys.platform == "win32":
            try:
                return pipe_stdout.read(size)
            except Exception:
                return None

        ready, _, _ = select.select([pipe_stdout], [], [], timeout)
        if ready:
            try:
                return pipe_stdout.read(size)
            except Exception:
                return None
        return None

    def get_stats(self) -> dict[str, float | int]:
        if self.last_frame_time is None:
            return {}

        return {
            "frame_count": self.frame_count,
            "time_since_last_frame": time.time() - self.last_frame_time,
        }

    def close(self) -> None:
        print(f"[INFO] Closing stream: {self.camera_id or self.rtsp_url}")
        if self.pipe:
            try:
                self.pipe.terminate()
                time.sleep(0.5)
                if self.pipe.poll() is None:
                    self.pipe.kill()
            except Exception:
                pass
            self.pipe = None
