#!/usr/bin/env python3
"""
Capture one Raspberry Pi camera image and sync it to another machine with scp.

Example:
    python cam.py --scp-target dhruv@192.168.1.20:/Users/dhruv/Desktop/dir

You can also set CAM_SCP_TARGET and then run:
    python cam.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one image on a Raspberry Pi and copy it with scp."
    )
    parser.add_argument(
        "--output-dir",
        default="captures",
        help="Directory on the Raspberry Pi where the image is saved first.",
    )
    parser.add_argument(
        "--filename",
        help="Image filename. Defaults to capture_YYYYmmdd_HHMMSS.jpg.",
    )
    parser.add_argument(
        "--scp-target",
        default=os.environ.get("CAM_SCP_TARGET"),
        help="Destination like user@host:/absolute/path/dir. Can also use CAM_SCP_TARGET.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture height in pixels.",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Seconds to let the camera auto-exposure settle before capture.",
    )
    parser.add_argument(
        "--no-scp",
        action="store_true",
        help="Only save the image on the Raspberry Pi; do not sync it.",
    )
    return parser.parse_args()


def build_output_path(output_dir: str, filename: str | None) -> Path:
    capture_dir = Path(output_dir).expanduser()
    capture_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{stamp}.jpg"

    return capture_dir / filename


def capture_with_picamera2(path: Path, width: int, height: int, warmup: float) -> None:
    from picamera2 import Picamera2

    camera = Picamera2()
    started = False
    try:
        config = camera.create_still_configuration(main={"size": (width, height)})
        camera.configure(config)
        camera.start()
        started = True
        time.sleep(warmup)
        camera.capture_file(str(path))
    finally:
        if started:
            camera.stop()
        camera.close()


def capture_with_libcamera(path: Path, width: int, height: int, warmup: float) -> None:
    camera_command = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if not camera_command:
        raise RuntimeError("Neither picamera2, rpicam-still, nor libcamera-still is available.")

    timeout_ms = max(1, int(warmup * 1000))
    command = [
        camera_command,
        "--width",
        str(width),
        "--height",
        str(height),
        "--timeout",
        str(timeout_ms),
        "--output",
        str(path),
    ]
    subprocess.run(command, check=True)


def capture_image(path: Path, width: int, height: int, warmup: float) -> None:
    try:
        capture_with_picamera2(path, width, height, warmup)
    except Exception as exc:
        print(f"Picamera2 failed ({exc}); falling back to rpicam-still/libcamera-still.")
        capture_with_libcamera(path, width, height, warmup)


def scp_image(path: Path, target: str) -> None:
    target = target.rstrip("/")
    subprocess.run(["scp", str(path), f"{target}/"], check=True)


def main() -> int:
    args = parse_args()
    output_path = build_output_path(args.output_dir, args.filename)

    try:
        capture_image(output_path, args.width, args.height, args.warmup)
        print(f"Saved image: {output_path}")

        if args.no_scp:
            return 0

        if not args.scp_target:
            print(
                "No scp target set. Pass --scp-target user@host:/path/to/dir "
                "or set CAM_SCP_TARGET.",
                file=sys.stderr,
            )
            return 2

        scp_image(output_path, args.scp_target)
        print(f"Synced image to: {args.scp_target}/")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(f"Capture/sync failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
