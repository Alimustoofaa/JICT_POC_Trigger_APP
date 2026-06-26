from __future__ import annotations

import argparse

import cv2

from src.camera import CameraError, get_camera
from src.camera.hikvision import FFmpegRTSPCamera


DEFAULT_CAMERA_ID = "trigger"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 15
DEFAULT_READ_TIMEOUT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display a configured RTSP camera.")
    parser.add_argument(
        "--camera-id",
        default=DEFAULT_CAMERA_ID,
        help=f"Camera id from config/cameras.json (default: {DEFAULT_CAMERA_ID})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Decoded frame width (default: {DEFAULT_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Decoded frame height (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Decoded frame rate (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_READ_TIMEOUT,
        help=f"Read timeout in seconds (default: {DEFAULT_READ_TIMEOUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = get_camera(args.camera_id)
    rtsp_query = {"videocodec": "h264"} if camera.brand == "AXIS" else {}
    rtsp_url = camera.get_rtsp_url(rtsp_query)

    stream = FFmpegRTSPCamera(
        rtsp_url=rtsp_url,
        camera_id=camera.camera_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        read_timeout=args.timeout,
    )

    try:
        stream.open()
        print(f"[INFO] Camera: {camera.name} ({camera.camera_id})")
        print(f"[INFO] RTSP: {rtsp_url}")
        print("[INFO] Press 'q' to quit.")

        while True:
            frame = stream.read()
            if frame is None:
                stats = stream.get_stats()
                print(f"[WARNING] Failed to read frame. Stats: {stats}")
                continue

            cv2.imshow(f"Camera Stream - {camera.camera_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Stopping stream...")
    finally:
        stream.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except CameraError as error:
        print(f"Camera error: {error}")
        if error.details:
            print(error.details)
