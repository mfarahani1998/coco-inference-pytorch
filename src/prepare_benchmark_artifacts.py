#!/usr/bin/env python3
"""
Convenience wrapper that prepares reusable benchmark artifacts.

Typical workflows:

Host machine, portable exports only:
  python prepare_benchmark_artifacts.py --framework ultralytics --portable-only --device auto
  python prepare_benchmark_artifacts.py --framework torchvision --portable-only --device auto

Target machine, TensorRT only from prepared ONNX:
  python prepare_benchmark_artifacts.py --framework ultralytics --formats trt --device 0

Notes:
- Ultralytics uses the revised two-mode flow from bench_ultralytics.py.
- Torchvision is kept compatible with its existing prepare flow for now.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


ULTRALYTICS_SUPPORTED = {
    "yolo26n",
    "yolo26s",
    "yolo26m",
    "yolo26l",
    "yolo26x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
    "yolov5nu",
    "yolov5mu",
    "yolov5xu",
}
TORCHVISION_SUPPORTED = {
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "maskrcnn_resnet50_fpn",
}
PORTABLE_FORMATS = {"torchscript", "onnx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare reusable benchmark artifacts for Ultralytics and torchvision benchmark scripts."
    )
    parser.add_argument(
        "--framework",
        choices=["ultralytics", "torchvision", "all"],
        default="all",
        help="Which benchmark family to prepare artifacts for.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the model list passed to the target script(s).",
    )
    parser.add_argument(
        "--imgsz", nargs="+", type=int, default=None, help="Override image sizes."
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help=(
            "Formats to prepare. If omitted, portable formats are prepared by default. "
            "TensorRT is typically built on the target machine."
        ),
    )
    parser.add_argument(
        "--portable-only",
        action="store_true",
        help="Shorthand for --formats torchscript onnx.",
    )
    parser.add_argument(
        "--include-half",
        dest="include_half",
        action="store_true",
        default=True,
        help="Prepare FP16 artifacts too when the selected device supports them.",
    )
    parser.add_argument("--no-include-half", dest="include_half", action="store_false")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data", type=Path, default=Path("configs/coco.yaml"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("."),
        help="Parent directory that will receive bench_cache_<framework>/ folders.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: List[str], dry_run: bool) -> None:
    print(format_cmd(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()

    here = Path(__file__).expanduser().resolve().parent
    ultra_script = here / "bench_ultralytics.py"
    tv_script = here / "bench_torchvision.py"

    if not ultra_script.exists():
        raise SystemExit("Missing script: %s" % ultra_script)
    if not tv_script.exists():
        raise SystemExit("Missing script: %s" % tv_script)

    formats = args.formats
    if args.portable_only:
        formats = ["torchscript", "onnx"]
    if not formats:
        formats = ["torchscript", "onnx"]

    frameworks = (
        [args.framework] if args.framework != "all" else ["ultralytics", "torchvision"]
    )

    for framework in frameworks:
        script = ultra_script if framework == "ultralytics" else tv_script
        framework_root = args.artifact_root.expanduser().resolve() / (
            "bench_cache_" + framework
        )

        if framework == "ultralytics":
            supported = ULTRALYTICS_SUPPORTED
        else:
            supported = TORCHVISION_SUPPORTED

        selected_models = None
        if args.models:
            selected_models = [model for model in args.models if model in supported]
            if args.framework != "all":
                invalid = [model for model in args.models if model not in supported]
                if invalid:
                    raise SystemExit(
                        "Unsupported model(s) for %s: %s"
                        % (framework, ", ".join(invalid))
                    )
            if not selected_models:
                print("[SKIP] No models selected for framework=%s" % framework)
                continue

        cmd: List[str] = [sys.executable, str(script)]

        if framework == "ultralytics":
            only_trt = set(formats) == {"trt"}
            mode = "prepared" if only_trt else "full"
            cmd.extend(
                [
                    "--mode",
                    mode,
                    "--prepare-only",
                    "--artifact-root",
                    str(framework_root),
                    "--device",
                    str(args.device),
                    "--data",
                    str(args.data),
                    "--workers",
                    str(args.workers),
                    "--formats",
                    *list(formats),
                    "--weights-dir",
                    str(args.weights_dir),
                ]
            )
        else:
            cmd.extend(
                [
                    "--mode",
                    "prepare",
                    "--artifact-root",
                    str(framework_root),
                    "--device",
                    str(args.device),
                    "--data",
                    str(args.data),
                    "--workers",
                    str(args.workers),
                    "--formats",
                    *list(formats),
                ]
            )

        if selected_models:
            cmd.extend(["--models", *selected_models])
        if args.imgsz:
            cmd.extend(["--imgsz", *[str(v) for v in args.imgsz]])
        if args.include_half:
            cmd.append("--include-half")
        else:
            cmd.append("--no-include-half")
        if args.rebuild_artifacts:
            cmd.append("--rebuild-artifacts")

        run_command(cmd, args.dry_run)


if __name__ == "__main__":
    main()
