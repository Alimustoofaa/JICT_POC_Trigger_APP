from __future__ import annotations

from datetime import datetime
from pathlib import Path
import select
import subprocess as sp
import sys
import time

import cv2
import numpy as np


LEFT_CAMERAS = {
    "left_1": {
        "ip": "192.168.100.19",
        "username": "admin",
        "password": "Halotec100",
        "stream_endpoint": "/axis-media/media.amp",
        "width": 640,
        "height": 480,
        "fps": 10,
        "read_timeout": 3,
    },
    "left_2": {
        "ip": "192.168.100.20",
        "username": "admin",
        "password": "Halotec100",
        "stream_endpoint": "/axis-media/media.amp",
        "width": 640,
        "height": 480,
        "fps": 10,
        "read_timeout": 3,
    },
}

LEFT_CAMERA_IDS = ["left_1", "left_2"]
CAPTURE_DIR = Path("captures/left")


def build_rtsp_url(camera_config: dict[str, str | int]) -> str:
    username = camera_config["username"]
    password = camera_config["password"]
    ip = camera_config["ip"]
    stream_endpoint = camera_config["stream_endpoint"]
    return f"rtsp://{username}:{password}@{ip}{stream_endpoint}"


class FFmpegRTSPCamera:
    def __init__(
        self,
        rtsp_url: str,
        camera_id: str,
        width: int,
        height: int,
        fps: int,
        read_timeout: int,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.read_timeout = read_timeout
        self.frame_bytes = self.width * self.height * 3
        self.pipe: sp.Popen[bytes] | None = None

        self.command = [
            "ffmpeg",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-analyzeduration",
            "0",
            "-probesize",
            "32",
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
        print(f"[INFO] Opening RTSP stream: {self.camera_id}")
        self.pipe = sp.Popen(
            self.command,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            bufsize=10**8,
        )

    def read(self) -> np.ndarray | None:
        if self.pipe is None or self.pipe.stdout is None:
            return None

        raw = self._read_with_timeout(self.pipe.stdout, self.frame_bytes, self.read_timeout)
        if raw is None or len(raw) != self.frame_bytes:
            return None

        return np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))

    def _read_with_timeout(self, pipe_stdout, size: int, timeout: int) -> bytes | None:
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

    def close(self) -> None:
        print(f"[INFO] Closing stream: {self.camera_id}")
        if self.pipe:
            try:
                self.pipe.terminate()
                time.sleep(0.5)
                if self.pipe.poll() is None:
                    self.pipe.kill()
            except Exception:
                pass
            self.pipe = None


class LeftImageCollector:
    def __init__(self) -> None:
        self.streams: dict[str, FFmpegRTSPCamera] = {}

    def open_cameras(self) -> None:
        for camera_id in LEFT_CAMERA_IDS:
            camera = LEFT_CAMERAS[camera_id]
            stream = FFmpegRTSPCamera(
                rtsp_url=build_rtsp_url(camera),
                camera_id=camera_id,
                width=int(camera["width"]),
                height=int(camera["height"]),
                fps=int(camera["fps"]),
                read_timeout=int(camera["read_timeout"]),
            )
            stream.open()
            self.streams[camera_id] = stream

    def close_cameras(self) -> None:
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
        cv2.destroyAllWindows()

    def read_frames(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera_id, stream in self.streams.items():
            frame = stream.read()
            if frame is not None:
                frames[camera_id] = frame
        return frames

    def merge_frames(self, frames: dict[str, np.ndarray]) -> np.ndarray | None:
        if not frames:
            return None

        base_frame = next(iter(frames.values()))
        frame_height, frame_width = base_frame.shape[:2]
        merged_frames = []

        for camera_id in LEFT_CAMERA_IDS:
            frame = frames.get(camera_id)
            if frame is None:
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            merged_frames.append(frame)

        return np.hstack(merged_frames)

    def save_capture(self, frame: np.ndarray) -> Path:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = CAPTURE_DIR / f"left_{timestamp}.jpg"
        cv2.imwrite(str(output_path), frame)
        return output_path

    def run(self) -> None:
        self.open_cameras()
        print("Press 'c' to capture image, 'q' to quit.")

        try:
            while True:
                frames = self.read_frames()
                merged_frame = self.merge_frames(frames)
                if merged_frame is None:
                    continue

                cv2.imshow("Left Camera Collector", merged_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    output_path = self.save_capture(merged_frame)
                    print(f"Captured: {output_path}")
                elif key == ord("q"):
                    break
        finally:
            self.close_cameras()


if __name__ == "__main__":
    LeftImageCollector().run()
