#!/usr/bin/env python3
"""
Ultralytics benchmark runner with round-robin repeats and reusable artifacts.

Design goals:
- two benchmark modes:
  * full: prepare missing portable artifacts (TorchScript/ONNX) and benchmark
  * prepared: use prepared portable artifacts and still build TensorRT locally if needed
- no artifact is exported more than once per benchmarking run
- artifacts are saved in per-model / per-image-size / per-precision / per-format folders
- resume support via CSV row matching and cached metric reuse
- repeat ordering is round-robin across the full experiment plan
- focus on Ultralytics models only

Notes:
- Portable artifacts are PyTorch weights, TorchScript, and ONNX.
- TensorRT engines are treated as local artifacts and are built from the cached ONNX export.
- A small sidecar metadata file is stored next to each artifact to validate reuse.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import shutil
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# NOTE: keep stdlib imports above third-party imports; Path/typing are used by metadata helpers.
from dataclasses import dataclass
import torch

DEFAULT_MODELS = [
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
]
DEFAULT_SIZES = [384, 512, 640, 768]
DEFAULT_FORMATS = ["pytorch", "torchscript", "onnx", "trt"]
PORTABLE_FORMATS = {"pytorch", "torchscript", "onnx"}
FORMAT_TO_EXPORT = {
    "pytorch": "-",
    "torchscript": "torchscript",
    "onnx": "onnx",
    "trt": "engine",
}
FORMAT_TO_NAME = {
    "pytorch": "PyTorch",
    "torchscript": "TorchScript",
    "onnx": "ONNX",
    "trt": "TensorRT",
}
NAME_TO_FORMAT = {value: key for key, value in FORMAT_TO_NAME.items()}
FORMAT_ORDER = {name: index for index, name in enumerate(DEFAULT_FORMATS)}
FIELDNAMES = [
    "framework",
    "model",
    "weights",
    "imgsz",
    "batch",
    "device",
    "repeat",
    "half",
    "precision",
    "format_name",
    "format_arg",
    "backend",
    "runtime_provider",
    "benchmark_impl",
    "input_geometry",
    "resize_mode",
    "conf",
    "max_det",
    "params",
    "gflops",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "ap50_95",
    "fps",
    "status",
    "error",
    "artifact_path",
    "artifact_size_mb",
    "save_dir",
    "run_name",
    "metric_key",
    "benchmark_wall_time_s",
]
METRIC_KEY = "metrics/mAP50-95(B)"
ARTIFACT_SCHEMA_VERSION = 2
BENCHMARK_IMPL = "ultralytics_round_robin_artifacts_v6_model_stats"
MODEL_STATS_SCHEMA_VERSION = 1
MODEL_STATS_IMPL = "ultralytics_fused_pytorch_model_stats_v1"
IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ArtifactKey:
    model: str
    imgsz: int
    format_key: str
    half: bool


@dataclass(frozen=True)
class ExperimentSpec:
    key: ArtifactKey
    weights_ref: str


@dataclass(frozen=True)
class RowKey:
    framework: str
    model: str
    weights: str
    imgsz: int
    batch: int
    device: str
    repeat: int
    half: int
    precision: str
    format_name: str
    conf: float
    max_det: int


@dataclass(frozen=True)
class MetricLookupKey:
    model: str
    weights: str
    imgsz: int
    half: int
    format_name: str
    conf: float
    max_det: int


@dataclass(frozen=True)
class ModelStatsKey:
    model: str
    weights: str
    imgsz: int


@dataclass
class ArtifactInfo:
    key: ArtifactKey
    artifact_dir: Path
    artifact_path: str
    artifact_size_mb: Optional[float]
    weights_ref: str


@dataclass
class PredictPassResult:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    wall_time_s: float
    num_images: int


@dataclass
class ValResult(PredictPassResult):
    ap50_95: float
    metric_key: str


@dataclass
class SourceBundle:
    speed_source: Union[str, List[str]]
    source_ref: str
    warmup_images: List[str]


@dataclass
class MetricCacheEntry:
    ap50_95: float
    metric_key: str


@dataclass
class ModelStatsEntry:
    params: Optional[int]
    gflops: Optional[float]


@lru_cache(maxsize=1)
def is_jetson_host() -> bool:
    machine = platform.machine().strip().lower()
    if machine not in {"aarch64", "arm64"}:
        return False
    if Path("/etc/nv_tegra_release").exists():
        return True
    for marker in (
        Path("/sys/devices/soc0/family"),
        Path("/sys/firmware/devicetree/base/model"),
    ):
        try:
            text = marker.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if "jetson" in text or "tegra" in text or "nvidia" in text:
            return True
    plat = platform.platform().lower()
    return "jetson" in plat or "tegra" in plat


def default_worker_count() -> int:
    if is_jetson_host():
        return 0
    return min(4, max(1, os.cpu_count() or 1))


def gc_cuda_barrier(device: str) -> None:
    gc.collect()
    if torch.cuda.is_available() and is_cuda_like_device(device):
        index = device_index_from_string(device)
        try:
            torch.cuda.synchronize(index)
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize(index)
        except Exception:
            pass
    gc.collect()


def _best_effort_close(value: Any) -> None:
    if value is None:
        return
    for method_name in ("close", "release"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _best_effort_setattr(obj: Any, name: str, value: Any) -> None:
    if obj is None or not hasattr(obj, name):
        return
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def release_autobackend_resources(autobackend: Any) -> None:
    if autobackend is None:
        return

    _best_effort_close(autobackend)
    backend = getattr(autobackend, "backend", None)
    for owner in (autobackend, backend):
        if owner is None:
            continue
        _best_effort_close(owner)
        for attr_name in (
            "session",
            "engine",
            "context",
            "stream",
            "cuda_graph",
            "io_binding",
            "bindings",
            "binding_addrs",
            "input_names",
            "output_names",
            "model",
        ):
            value = getattr(owner, attr_name, None)
            _best_effort_close(value)
            _best_effort_setattr(owner, attr_name, None)
    _best_effort_setattr(autobackend, "backend", None)


def release_predictor_resources(predictor: Any) -> None:
    if predictor is None:
        return

    vid_writer = getattr(predictor, "vid_writer", None)
    if isinstance(vid_writer, dict):
        for writer in vid_writer.values():
            _best_effort_close(writer)

    release_autobackend_resources(getattr(predictor, "model", None))
    for attr_name in (
        "results",
        "batch",
        "dataset",
        "plotted_img",
        "source_type",
        "vid_writer",
        "windows",
        "model",
    ):
        _best_effort_setattr(predictor, attr_name, None)


def release_yolo_runtime(runner: Any, device: str) -> None:
    try:
        if runner is not None:
            release_predictor_resources(getattr(runner, "predictor", None))
            release_autobackend_resources(getattr(runner, "model", None))
            for attr_name in ("predictor", "trainer", "metrics", "session", "model"):
                _best_effort_setattr(runner, attr_name, None)
    finally:
        gc_cuda_barrier(device)


def run_validation_isolated(
    YOLO: Any,
    runtime_path: str,
    key: ArtifactKey,
    args: argparse.Namespace,
) -> ValResult:
    runner = None
    try:
        runner = YOLO(runtime_path)
        return run_validation(runner, key, args)
    finally:
        release_yolo_runtime(runner, args.device)


def run_predict_speed_isolated(
    YOLO: Any,
    runtime_path: str,
    key: ArtifactKey,
    source_bundle: SourceBundle,
    args: argparse.Namespace,
) -> PredictPassResult:
    runner = None
    try:
        runner = YOLO(runtime_path)
        warmup_model(runner, key, source_bundle, args)
        return run_predict_speed(runner, key, source_bundle, args)
    finally:
        release_yolo_runtime(runner, args.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Ultralytics models with reusable exports, round-robin repeats, "
            "resume support, and target-local TensorRT preparation."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Ultralytics model names to benchmark. RT-DETR and YOLO-NAS are skipped by default.",
    )
    parser.add_argument("--imgsz", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=sorted(FORMAT_TO_EXPORT.keys()),
        help="Benchmark formats. TensorRT is built from cached ONNX on the local machine.",
    )
    parser.add_argument(
        "--mode",
        default="full",
        help=(
            "Benchmark mode. Canonical values are 'full' and 'prepared'. "
            "Legacy aliases: 'benchmark' -> 'prepared', 'prepare' -> 'prepared' + --prepare-only."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare the required artifacts and exit without benchmarking.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--include-half",
        dest="include_half",
        action="store_true",
        default=True,
        help="Include FP16 experiments when the selected device supports them.",
    )
    parser.add_argument("--no-include-half", dest="include_half", action="store_false")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/coco.yaml"),
        help="Dataset YAML for validation and predict source resolution.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory containing local .pt weights. Falls back to '<model>.pt' if not found.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="Ultralytics device, e.g. auto, 0, cuda:0, cpu, mps, or dla:0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("bench_cache_ultralytics"),
        help="Root directory for reusable exports, target-local TensorRT engines, and cached metrics.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("benchmark_ultralytics.csv"),
    )
    parser.add_argument("--score-thr", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument(
        "--eval-policy",
        choices=["once", "every-repeat"],
        default="once",
        help="once: validate each artifact once and reuse its AP for all repeats.",
    )
    parser.add_argument(
        "--reuse-accuracy-cache",
        dest="reuse_accuracy_cache",
        action="store_true",
        default=True,
        help="Reuse metrics.json when the artifact and evaluation settings still match.",
    )
    parser.add_argument(
        "--no-reuse-accuracy-cache", dest="reuse_accuracy_cache", action="store_false"
    )
    parser.add_argument(
        "--measure-model-stats",
        dest="measure_model_stats",
        action="store_true",
        default=True,
        help=(
            "Measure fused-model parameter counts and GFLOPs once per model/image size and "
            "reuse them across PyTorch, TorchScript, ONNX, TensorRT, FP32, and FP16 rows."
        ),
    )
    parser.add_argument(
        "--no-measure-model-stats",
        dest="measure_model_stats",
        action="store_false",
    )
    parser.add_argument(
        "--reuse-model-stats-cache",
        dest="reuse_model_stats_cache",
        action="store_true",
        default=True,
        help=(
            "Reuse model_stats.json when the weights signature and image size still match. "
            "This avoids re-measuring params/GFLOPs on resumed runs and prepared artifacts."
        ),
    )
    parser.add_argument(
        "--no-reuse-model-stats-cache",
        dest="reuse_model_stats_cache",
        action="store_false",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume by skipping already-successful rows that match the current experiment signature.",
    )
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--skip-existing",
        dest="resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="resume",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Force regeneration of exported artifacts and TensorRT engines.",
    )
    parser.add_argument(
        "--warmup-images",
        type=int,
        default=8,
        help="Number of images used for one warmup predict() pass after loading a runtime backend.",
    )
    parser.add_argument(
        "--trt-workspace",
        type=float,
        default=None,
        help="Optional TensorRT workspace size in GiB when building local engines from ONNX.",
    )
    parser.add_argument(
        "--trust-existing-trt",
        action="store_true",
        help=(
            "Use an existing .engine even if it lacks local-build metadata or its metadata does not match "
            "the current machine. Disabled by default because TensorRT engines are target-specific."
        ),
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Keep Ultralytics progress output enabled.",
    )
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()

    mode_raw = str(args.mode).strip().lower()
    if mode_raw == "benchmark":
        args.mode = "prepared"
    elif mode_raw == "prepare":
        args.mode = "prepared"
        args.prepare_only = True
    elif mode_raw in {"full", "prepared"}:
        args.mode = mode_raw
    else:
        raise SystemExit(
            "Unsupported --mode value: %s (use 'full' or 'prepared')" % args.mode
        )

    args.device = normalize_ultralytics_device(args.device)
    args.weights_dir = args.weights_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.out_csv = args.out_csv.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    return args


def normalize_ultralytics_device(device: Any) -> str:
    text = str(device).strip() if device is not None else "auto"
    if not text or text.lower() == "auto":
        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    low = text.lower()
    if low == "cuda":
        return "0"
    if low.startswith("cuda:"):
        tail = low.split(":", 1)[1]
        return tail if tail.isdigit() else text
    return text


def is_cuda_like_device(device: str) -> bool:
    text = str(device).strip().lower()
    return text.isdigit() or text.startswith("cuda") or text.startswith("dla:")


def device_supports_half(device: str) -> bool:
    return is_cuda_like_device(device)


def device_supports_trt(device: str) -> bool:
    return is_cuda_like_device(device)


def device_index_from_string(device: str) -> int:
    text = str(device).strip().lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:"):
        tail = text.split(":", 1)[1]
        if tail.isdigit():
            return int(tail)
    return 0


def trt_dla_core_from_device(device: str) -> Optional[int]:
    text = str(device).strip().lower()
    if text.startswith("dla:"):
        tail = text.split(":", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def is_excluded_model(name: str) -> bool:
    low = name.lower()
    return "rtdetr" in low or "rt-detr" in low or "yolo_nas" in low or "yolo-nas" in low


def precision_name(half: bool) -> str:
    return "fp16" if half else "fp32"


def run_name_for(key: ArtifactKey, repeat: int) -> str:
    return "{model}_img{imgsz}_{fmt}_{prec}_r{repeat}".format(
        model=key.model,
        imgsz=key.imgsz,
        fmt=key.format_key,
        prec=precision_name(key.half),
        repeat=repeat,
    )


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except Exception:
        return None


def printable(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return "%.4f" % value
    return str(value)


def rounded_float(value: Any, ndigits: int = 6) -> float:
    parsed = parse_float(value)
    if parsed is None:
        return float("nan")
    return round(float(parsed), ndigits)


def resolve_model_reference(model_name: str, weights_dir: Path) -> str:
    candidate = (weights_dir / f"{model_name}.pt").expanduser()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return str(candidate.resolve())


def file_size_mb(path_str: str) -> Optional[float]:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return float(path.stat().st_size) / (1024.0 * 1024.0)
    except Exception:
        return None


@lru_cache(maxsize=None)
def _sha256_for_file(path_str: str, size_bytes: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path_str).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path_like: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Return a content-based signature that survives copies and directory moves.

    Earlier revisions stored absolute paths and mtimes in artifact metadata. That made
    prepared portable artifacts appear invalid after being copied to another machine or
    relocated inside a different artifact root. The new signature uses the file name,
    size, and SHA-256 content hash instead.
    """
    path = Path(path_like).expanduser()
    if not path.exists() or not path.is_file():
        return None
    try:
        resolved = path.resolve()
        stat = resolved.stat()
    except Exception:
        return None
    return {
        "name": resolved.name,
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_for_file(
            str(resolved),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        ),
    }


def weights_signature(weights_ref: str) -> Dict[str, Any]:
    """Return path-independent weight provenance for artifact metadata.

    The stored basename is portable across machines, while the optional file signature
    keeps strong provenance when the source .pt file is present locally.
    """
    path = Path(str(weights_ref)).expanduser()
    signature: Dict[str, Any] = {"name": path.name or str(weights_ref)}
    file_sig = file_signature(path)
    if file_sig is not None:
        signature["file"] = file_sig
    return signature


def _path_name_from_value(value: Any) -> str:
    text = safe_str(value).strip()
    return Path(text).name if text else ""


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value)))
        except Exception:
            return None


def file_signature_matches(stored_signature: Any, path_like: Union[str, Path]) -> bool:
    """Compare current file content against either v2 or legacy v1 metadata.

    v2 signatures are content-based. Legacy v1 signatures stored path + size + mtime;
    for those we keep a conservative compatibility path by matching file size and the
    basename encoded in the old path.
    """
    if not isinstance(stored_signature, dict):
        return False
    current = file_signature(path_like)
    if current is None:
        return False

    stored_sha = safe_str(stored_signature.get("sha256"))
    if stored_sha:
        stored_size = _int_or_none(stored_signature.get("size_bytes"))
        if stored_size is not None and stored_size != int(current["size_bytes"]):
            return False
        stored_name = safe_str(stored_signature.get("name"))
        return stored_sha == safe_str(current.get("sha256")) and (
            not stored_name or stored_name == safe_str(current.get("name"))
        )

    stored_size = _int_or_none(stored_signature.get("size_bytes"))
    if stored_size is None or stored_size != int(current["size_bytes"]):
        return False
    stored_name = _path_name_from_value(stored_signature.get("path")) or safe_str(
        stored_signature.get("name")
    )
    current_name = safe_str(current.get("name"))
    return not stored_name or stored_name == current_name


def weights_signature_matches(stored_signature: Any, weights_ref: str) -> bool:
    if not isinstance(stored_signature, dict):
        return False

    current = weights_signature(weights_ref)
    stored_name = safe_str(stored_signature.get("name")) or _path_name_from_value(
        stored_signature.get("weights_ref")
    )
    current_name = safe_str(current.get("name"))
    if stored_name and current_name and stored_name != current_name:
        return False

    stored_file_sig = stored_signature.get("file")
    if stored_file_sig is None:
        return bool(stored_name) and stored_name == current_name
    return file_signature_matches(stored_file_sig, weights_ref)


def try_ultralytics_version() -> Optional[str]:
    try:
        import ultralytics

        return str(ultralytics.__version__)
    except Exception:
        return None


def try_tensorrt_version() -> Optional[str]:
    try:
        import tensorrt as trt  # type: ignore

        return str(trt.__version__)
    except Exception:
        return None


def current_trt_system_signature(device: str) -> Dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available() and is_cuda_like_device(device):
        try:
            gpu_name = torch.cuda.get_device_name(device_index_from_string(device))
        except Exception:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = None
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "tensorrt": try_tensorrt_version(),
        "device": str(device),
        "gpu_name": gpu_name,
    }


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def ensure_csv_schema(out_csv: Path) -> None:
    if not out_csv.exists():
        return
    try:
        with out_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            current_header = reader.fieldnames or []
            rows = list(reader)
    except Exception:
        return
    if current_header == FIELDNAMES:
        return
    if set(current_header).issubset(set(FIELDNAMES)):
        upgraded_rows = [
            {field: row.get(field, "") for field in FIELDNAMES} for row in rows
        ]
        temp_path = out_csv.with_name(
            out_csv.stem + ".schema_upgrade_" + str(int(time.time())) + out_csv.suffix
        )
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in upgraded_rows:
                writer.writerow(row)
        temp_path.replace(out_csv)
        print(
            "[INFO] Existing CSV header was upgraded in-place to the current schema: %s"
            % out_csv
        )
        return
    backup = out_csv.with_name(
        out_csv.stem + ".legacy_" + str(int(time.time())) + out_csv.suffix
    )
    out_csv.rename(backup)
    print(
        "[INFO] Existing CSV header differs from the current schema. Moved old file to: %s"
        % backup
    )


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_schema(out_csv)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def row_key_from_row(row: Dict[str, Any]) -> Optional[RowKey]:
    try:
        return RowKey(
            framework=safe_str(row.get("framework")),
            model=safe_str(row.get("model")),
            weights=safe_str(row.get("weights")),
            imgsz=int(float(safe_str(row.get("imgsz")))),
            batch=int(float(safe_str(row.get("batch")))),
            device=safe_str(row.get("device")),
            repeat=int(float(safe_str(row.get("repeat")))),
            half=int(float(safe_str(row.get("half")) or "0")),
            precision=safe_str(row.get("precision")),
            format_name=safe_str(row.get("format_name")),
            conf=rounded_float(row.get("conf")),
            max_det=int(float(safe_str(row.get("max_det")) or "0")),
        )
    except Exception:
        return None


def metric_lookup_key_from_row(row: Dict[str, Any]) -> Optional[MetricLookupKey]:
    try:
        return MetricLookupKey(
            model=safe_str(row.get("model")),
            weights=safe_str(row.get("weights")),
            imgsz=int(float(safe_str(row.get("imgsz")))),
            half=int(float(safe_str(row.get("half")) or "0")),
            format_name=safe_str(row.get("format_name")),
            conf=rounded_float(row.get("conf")),
            max_det=int(float(safe_str(row.get("max_det")) or "0")),
        )
    except Exception:
        return None


def load_existing_success_keys(out_csv: Path) -> Set[RowKey]:
    keys: Set[RowKey] = set()
    if not out_csv.exists():
        return keys
    try:
        with out_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if safe_str(row.get("status")).strip().lower() != "success":
                    continue
                key = row_key_from_row(row)
                if key is not None:
                    keys.add(key)
    except Exception:
        return set()
    return keys


def load_existing_metric_rows(out_csv: Path) -> Dict[MetricLookupKey, MetricCacheEntry]:
    cached: Dict[MetricLookupKey, MetricCacheEntry] = {}
    if not out_csv.exists():
        return cached
    try:
        with out_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if safe_str(row.get("status")).strip().lower() != "success":
                    continue
                ap50_95 = parse_float(row.get("ap50_95"))
                if ap50_95 is None:
                    continue
                key = metric_lookup_key_from_row(row)
                if key is None:
                    continue
                cached[key] = MetricCacheEntry(
                    ap50_95=float(ap50_95),
                    metric_key=safe_str(row.get("metric_key")) or METRIC_KEY,
                )
    except Exception:
        return {}
    return cached


def model_stats_key_from_row(row: Dict[str, Any]) -> Optional[ModelStatsKey]:
    try:
        return ModelStatsKey(
            model=safe_str(row.get("model")),
            weights=safe_str(row.get("weights")),
            imgsz=int(float(safe_str(row.get("imgsz")))),
        )
    except Exception:
        return None


def load_existing_model_stats_rows(
    out_csv: Path,
) -> Dict[ModelStatsKey, ModelStatsEntry]:
    cached: Dict[ModelStatsKey, ModelStatsEntry] = {}
    if not out_csv.exists():
        return cached
    try:
        with out_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = model_stats_key_from_row(row)
                if key is None:
                    continue
                params = parse_float(row.get("params"))
                gflops = parse_float(row.get("gflops"))
                if params is None and gflops is None:
                    continue
                cached[key] = ModelStatsEntry(
                    params=(int(params) if params is not None else None),
                    gflops=(float(gflops) if gflops is not None else None),
                )
    except Exception:
        return {}
    return cached


def model_stats_lookup_key(
    key: ArtifactKey,
    weights_ref: str,
) -> ModelStatsKey:
    return ModelStatsKey(
        model=key.model,
        weights=weights_ref,
        imgsz=int(key.imgsz),
    )


def model_stats_json_path(args: argparse.Namespace, key: ModelStatsKey) -> Path:
    return (
        args.artifact_root / key.model / ("img%d" % int(key.imgsz)) / "model_stats.json"
    )


def build_model_stats_payload(
    key: ModelStatsKey,
    params: Optional[int],
    gflops: Optional[float],
) -> Dict[str, Any]:
    return {
        "schema_version": MODEL_STATS_SCHEMA_VERSION,
        "stats_impl": MODEL_STATS_IMPL,
        "model": key.model,
        "imgsz": int(key.imgsz),
        "weights_signature": weights_signature(key.weights),
        "params": int(params) if params is not None else None,
        "gflops": float(gflops) if gflops is not None else None,
        "shared_across_formats": True,
        "shared_across_precisions": True,
        "batch": 1,
        "ultralytics_version": try_ultralytics_version(),
    }


def model_stats_payload_matches(
    payload: Optional[Dict[str, Any]],
    key: ModelStatsKey,
) -> bool:
    if payload is None:
        return False
    return (
        _int_or_none(payload.get("schema_version")) == MODEL_STATS_SCHEMA_VERSION
        and payload.get("stats_impl") == MODEL_STATS_IMPL
        and payload.get("model") == key.model
        and _int_or_none(payload.get("imgsz")) == int(key.imgsz)
        and weights_signature_matches(payload.get("weights_signature"), key.weights)
    )


def load_cached_model_stats(
    args: argparse.Namespace,
    key: ModelStatsKey,
) -> Optional[ModelStatsEntry]:
    if not args.reuse_model_stats_cache:
        return None
    payload = load_json(model_stats_json_path(args, key))
    if not model_stats_payload_matches(payload, key):
        return None
    params = payload.get("params")  # type: ignore
    gflops = payload.get("gflops")  # type: ignore
    if params is None and gflops is None:
        return None
    return ModelStatsEntry(
        params=(int(params) if params is not None else None),
        gflops=(float(gflops) if gflops is not None else None),
    )


def save_cached_model_stats(
    args: argparse.Namespace,
    key: ModelStatsKey,
    entry: ModelStatsEntry,
) -> None:
    save_json(
        model_stats_json_path(args, key),
        build_model_stats_payload(
            key=key,
            params=entry.params,
            gflops=entry.gflops,
        ),
    )


def model_stats_entry_complete(entry: Optional[ModelStatsEntry]) -> bool:
    return bool(
        entry is not None and entry.params is not None and entry.gflops is not None
    )


def ensure_model_stats_imports() -> Tuple[Any, Any, Optional[Any]]:
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ultralytics.utils.torch_utils model stat helpers: %s"
            % exc
        )
    try:
        from ultralytics.utils.torch_utils import get_flops_with_torch_profiler
    except Exception:
        get_flops_with_torch_profiler = None
    return get_num_params, get_flops, get_flops_with_torch_profiler


def build_fused_model_stats_module(base_model: Any) -> Any:
    module = deepcopy(getattr(base_model, "model", base_model))
    try:
        module.to("cpu")
    except Exception:
        pass
    try:
        module.float()
    except Exception:
        pass
    try:
        module.eval()
    except Exception:
        pass
    try:
        for parameter in module.parameters():
            parameter.requires_grad = False
    except Exception:
        pass
    fuse = getattr(module, "fuse", None)
    if callable(fuse):
        try:
            module = fuse(verbose=False)
        except TypeError:
            module = fuse()
    return module


def measure_model_stats_from_fused_module(
    fused_model: Any,
    imgsz: int,
    get_num_params: Any,
    get_flops: Any,
    get_flops_with_torch_profiler: Optional[Any],
) -> ModelStatsEntry:
    params = int(get_num_params(fused_model))
    gflops = float(get_flops(fused_model, imgsz=imgsz))
    if gflops <= 0.0 and get_flops_with_torch_profiler is not None:
        gflops = float(get_flops_with_torch_profiler(fused_model, imgsz=imgsz))
    return ModelStatsEntry(
        params=params,
        gflops=(gflops if gflops > 0.0 else None),
    )


def prepare_model_stats_for_model(
    args: argparse.Namespace,
    model_name: str,
    weights_ref: str,
    imgsz_values: Sequence[int],
    get_base_model: Any,
    existing_model_stats: Dict[ModelStatsKey, ModelStatsEntry],
) -> Dict[ModelStatsKey, ModelStatsEntry]:
    prepared: Dict[ModelStatsKey, ModelStatsEntry] = {}
    fallback_entries: Dict[ModelStatsKey, ModelStatsEntry] = {}
    if not args.measure_model_stats:
        return prepared

    missing: List[ModelStatsKey] = []
    for imgsz in sorted({int(value) for value in imgsz_values}):
        key = ModelStatsKey(model=model_name, weights=weights_ref, imgsz=int(imgsz))
        entry = existing_model_stats.get(key)
        if entry is None:
            entry = load_cached_model_stats(args, key)
        if entry is not None:
            fallback_entries[key] = entry
        if model_stats_entry_complete(entry):
            prepared[key] = entry  # type: ignore
        else:
            missing.append(key)

    if not missing:
        return prepared

    try:
        base_model = get_base_model()
    except Exception as exc:
        print(
            "[WARN] Could not load weights for model stats %s: %s" % (model_name, exc)
        )
        prepared.update(
            {
                key: entry
                for key, entry in fallback_entries.items()
                if key not in prepared
            }
        )
        return prepared

    get_num_params = None
    get_flops = None
    get_flops_with_torch_profiler = None
    fused_model = None
    try:
        get_num_params, get_flops, get_flops_with_torch_profiler = (
            ensure_model_stats_imports()
        )
        fused_model = build_fused_model_stats_module(base_model)
        for key in missing:
            try:
                entry = measure_model_stats_from_fused_module(
                    fused_model=fused_model,
                    imgsz=key.imgsz,
                    get_num_params=get_num_params,
                    get_flops=get_flops,
                    get_flops_with_torch_profiler=get_flops_with_torch_profiler,
                )
                prepared[key] = entry
                save_cached_model_stats(args, key, entry)
            except Exception as exc:
                print(
                    "[WARN] Failed to measure model stats for %s imgsz=%s: %s"
                    % (key.model, key.imgsz, exc)
                )
    except Exception as exc:
        print(
            "[WARN] Failed to initialize model stat measurement for %s: %s"
            % (model_name, exc)
        )
    finally:
        fused_model = None
        gc.collect()
    prepared.update(
        {key: entry for key, entry in fallback_entries.items() if key not in prepared}
    )
    return prepared


def artifact_base_dir(args: argparse.Namespace, key: ArtifactKey) -> Path:
    return (
        args.artifact_root
        / key.model
        / ("img%d" % int(key.imgsz))
        / precision_name(key.half)
        / key.format_key
    )


def artifact_file_path(args: argparse.Namespace, key: ArtifactKey) -> Optional[Path]:
    base = artifact_base_dir(args, key)
    if key.format_key == "pytorch":
        return None
    if key.format_key == "torchscript":
        return base / (key.model + ".torchscript")
    if key.format_key == "onnx":
        return base / (key.model + ".onnx")
    if key.format_key == "trt":
        return base / (key.model + ".engine")
    return None


def artifact_metadata_path(args: argparse.Namespace, key: ArtifactKey) -> Path:
    return artifact_base_dir(args, key) / "artifact.json"


def metrics_json_path(args: argparse.Namespace, key: ArtifactKey) -> Path:
    return artifact_base_dir(args, key) / "metrics.json"


def resolve_data_root_and_val(data_yaml: Path) -> Tuple[Path, Path]:
    data_yaml = data_yaml.expanduser().resolve()
    with data_yaml.open("r") as handle:
        import yaml

        data = yaml.safe_load(handle)

    root = Path(str(data.get("path", ".")))
    if not root.is_absolute():
        candidates = [
            (data_yaml.parent / root).resolve(),
            (data_yaml.parent.parent / root).resolve(),
        ]
        chosen = None
        for candidate in candidates:
            if candidate.exists():
                chosen = candidate
                break
        root = chosen if chosen is not None else candidates[0]

    val = data.get("val")
    if val is None:
        raise KeyError("%s missing required key: val" % data_yaml)

    val_path = Path(str(val))
    if not val_path.is_absolute():
        val_path = (root / val_path).resolve()
    return root.resolve(), val_path.resolve()


def iter_image_list_from_val(root: Path, val_path: Path) -> List[str]:
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        images: List[str] = []
        with val_path.open("r") as handle:
            for line in handle:
                item = line.strip()
                if not item:
                    continue
                p = Path(item).expanduser()
                if not p.is_absolute():
                    candidates = [
                        (val_path.parent / p).resolve(),
                        (root / p).resolve(),
                    ]
                    resolved = None
                    for candidate in candidates:
                        if candidate.exists():
                            resolved = candidate
                            break
                    p = resolved if resolved is not None else candidates[0]
                images.append(str(p.resolve()))
        return images

    if val_path.is_dir():
        return [
            str(path.resolve())
            for path in sorted(val_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

    if val_path.is_file() and val_path.suffix.lower() in IMAGE_SUFFIXES:
        return [str(val_path.resolve())]

    raise FileNotFoundError("Could not resolve validation images from: %s" % val_path)


def build_source_bundle(data_yaml: Path, warmup_images: int) -> SourceBundle:
    root, val_path = resolve_data_root_and_val(data_yaml)
    image_list = iter_image_list_from_val(root, val_path)
    if not image_list:
        raise RuntimeError(
            "No validation images were found for predict() source resolution."
        )
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        speed_source: Union[str, List[str]] = str(val_path)
        source_ref = str(val_path)
    elif val_path.is_dir():
        speed_source = str(val_path)
        source_ref = str(val_path)
    else:
        speed_source = image_list
        source_ref = str(val_path)
    return SourceBundle(
        speed_source=speed_source,
        source_ref=source_ref,
        warmup_images=image_list[: max(0, int(warmup_images))],
    )


def export_path_to_target(exported: Union[str, Path], target: Path) -> Path:
    exported_path = Path(str(exported)).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_target = target.expanduser().resolve()
    if exported_path == resolved_target:
        return exported_path
    shutil.move(exported_path, resolved_target)
    return resolved_target


def ensure_yolo_import() -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("Failed to import ultralytics: %s" % exc)
    return YOLO


def ensure_trt_builder_import() -> Any:
    try:
        from ultralytics.utils.export.engine import onnx2engine
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ultralytics.utils.export.engine.onnx2engine: %s" % exc
        )
    return onnx2engine


def read_onnx_embedded_metadata(
    onnx_path: Union[str, Path],
) -> Optional[Dict[str, Any]]:
    """Read Ultralytics metadata embedded in an exported ONNX file.

    Ultralytics writes export metadata into ONNX metadata_props. Reusing that metadata when
    building TensorRT engines preserves task/batch/imgsz/class-name hints and avoids spurious
    runtime warnings when the engine is loaded later via AutoBackend.
    """
    try:
        import onnx
    except Exception:
        return None

    path = Path(onnx_path).expanduser().resolve()
    try:
        try:
            model = onnx.load(str(path), load_external_data=False)
        except TypeError:
            model = onnx.load(str(path))
    except Exception:
        return None

    metadata_props = getattr(model, "metadata_props", None) or []
    metadata: Dict[str, Any] = {}
    for item in metadata_props:
        key = getattr(item, "key", None)
        value = getattr(item, "value", None)
        if key:
            metadata[str(key)] = value
    return metadata or None


def build_portable_artifact_metadata(
    key: ArtifactKey,
    artifact_path: str,
    weights_ref: str,
    assumed_existing: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "portable": True,
        "format_key": key.format_key,
        "model": key.model,
        "imgsz": int(key.imgsz),
        "half": bool(key.half),
        "weights_signature": weights_signature(weights_ref),
        "artifact_signature": file_signature(artifact_path),
        "ultralytics_version": try_ultralytics_version(),
    }
    if assumed_existing:
        payload["assumed_existing"] = True
    return payload


def build_trt_artifact_metadata(
    key: ArtifactKey,
    artifact_path: str,
    weights_ref: str,
    onnx_path: Union[str, Path],
    device: str,
) -> Dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "portable": False,
        "format_key": key.format_key,
        "model": key.model,
        "imgsz": int(key.imgsz),
        "half": bool(key.half),
        "weights_signature": weights_signature(weights_ref),
        "artifact_signature": file_signature(artifact_path),
        "source_onnx_signature": file_signature(onnx_path),
        "built_on": current_trt_system_signature(device),
        "ultralytics_version": try_ultralytics_version(),
    }


def portable_artifact_metadata_matches(
    meta: Optional[Dict[str, Any]],
    key: ArtifactKey,
    target: Path,
    weights_ref: str,
    require_weights_signature: bool = True,
) -> bool:
    if meta is None:
        return False
    if not (
        meta.get("portable") is True
        and meta.get("format_key") == key.format_key
        and meta.get("model") == key.model
        and int(meta.get("imgsz", -1)) == int(key.imgsz)
        and bool(meta.get("half")) == bool(key.half)
    ):
        return False
    if not file_signature_matches(meta.get("artifact_signature"), target):
        return False
    if not require_weights_signature:
        return True
    stored_weights = meta.get("weights_signature")
    if meta.get("assumed_existing") and not (
        isinstance(stored_weights, dict) and stored_weights.get("file") is not None
    ):
        return False
    return weights_signature_matches(stored_weights, weights_ref)


def trt_artifact_metadata_matches(
    meta: Optional[Dict[str, Any]],
    key: ArtifactKey,
    target: Path,
    weights_ref: str,
    onnx_path: Path,
    device: str,
) -> bool:
    del weights_ref  # engine reuse is keyed on the local engine + source ONNX + system.
    if meta is None:
        return False
    return (
        meta.get("portable") is False
        and meta.get("format_key") == key.format_key
        and meta.get("model") == key.model
        and int(meta.get("imgsz", -1)) == int(key.imgsz)
        and bool(meta.get("half")) == bool(key.half)
        and file_signature_matches(meta.get("artifact_signature"), target)
        and file_signature_matches(meta.get("source_onnx_signature"), onnx_path)
        and meta.get("built_on") == current_trt_system_signature(device)
    )


def export_artifact(
    base_model: Any,
    key: ArtifactKey,
    args: argparse.Namespace,
    target: Path,
) -> Path:
    export_kwargs: Dict[str, Any] = {
        "format": FORMAT_TO_EXPORT[key.format_key],
        "imgsz": int(key.imgsz),
        "device": args.device,
        "batch": 1,
    }
    if key.half:
        export_kwargs["half"] = True
    try:
        exported = base_model.export(**export_kwargs)
    except Exception as exc:
        raise RuntimeError("Export failed for %s: %s" % (key.format_key, exc))
    return export_path_to_target(exported, target)


def build_tensorrt_engine_from_onnx(
    onnx_path: Path,
    engine_path: Path,
    key: ArtifactKey,
    args: argparse.Namespace,
) -> Path:
    onnx2engine = ensure_trt_builder_import()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    dla_core = trt_dla_core_from_device(args.device)
    if dla_core is not None and not key.half:
        raise RuntimeError(
            "TensorRT DLA export requires FP16 in this script. Re-run with --include-half and a FP16 TensorRT row."
        )

    trt_metadata = read_onnx_embedded_metadata(onnx_path)
    try:
        onnx2engine(
            onnx_file=str(onnx_path),
            engine_file=str(engine_path),
            workspace=args.trt_workspace,
            half=bool(key.half),
            int8=False,
            dynamic=False,
            shape=(1, 3, int(key.imgsz), int(key.imgsz)),
            dla=dla_core,
            metadata=trt_metadata,
            verbose=bool(args.progress),
            prefix="[TRT:%s:%s] " % (key.model, key.imgsz),
        )
    except Exception as exc:
        raise RuntimeError("TensorRT engine build failed: %s" % exc)
    if not engine_path.exists():
        raise RuntimeError("TensorRT engine was not created: %s" % engine_path)
    return engine_path.resolve()


def ensure_portable_artifact(
    key: ArtifactKey,
    args: argparse.Namespace,
    weights_ref: str,
    get_base_model: Optional[Any],
) -> ArtifactInfo:
    artifact_dir = artifact_base_dir(args, key)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if key.format_key == "pytorch":
        return ArtifactInfo(
            key=key,
            artifact_dir=artifact_dir,
            artifact_path=weights_ref,
            artifact_size_mb=file_size_mb(weights_ref),
            weights_ref=weights_ref,
        )

    target = artifact_file_path(args, key)
    if target is None:
        raise RuntimeError("Unsupported format key: %s" % key.format_key)

    meta_path = artifact_metadata_path(args, key)
    meta = load_json(meta_path)
    ready = (
        target.exists()
        and not args.rebuild_artifacts
        and portable_artifact_metadata_matches(
            meta,
            key,
            target,
            weights_ref,
            require_weights_signature=(args.mode != "prepared"),
        )
    )
    if ready:
        return ArtifactInfo(
            key=key,
            artifact_dir=artifact_dir,
            artifact_path=str(target),
            artifact_size_mb=file_size_mb(str(target)),
            weights_ref=weights_ref,
        )

    if target.exists() and not args.rebuild_artifacts and meta is None:
        save_json(
            meta_path,
            build_portable_artifact_metadata(
                key=key,
                artifact_path=str(target),
                weights_ref=weights_ref,
                assumed_existing=True,
            ),
        )
        return ArtifactInfo(
            key=key,
            artifact_dir=artifact_dir,
            artifact_path=str(target),
            artifact_size_mb=file_size_mb(str(target)),
            weights_ref=weights_ref,
        )

    if args.mode == "prepared":
        state = "metadata mismatch" if target.exists() else "missing"
        raise RuntimeError(
            "Prepared %s artifact is %s: %s" % (key.format_key, state, target)
        )

    if get_base_model is None:
        raise RuntimeError(
            "Artifact %s is missing and no base model loader is available." % target
        )

    base_model = get_base_model()
    copied = export_artifact(base_model, key, args, target)
    save_json(
        meta_path,
        build_portable_artifact_metadata(
            key=key,
            artifact_path=str(copied),
            weights_ref=weights_ref,
        ),
    )
    return ArtifactInfo(
        key=key,
        artifact_dir=artifact_dir,
        artifact_path=str(copied),
        artifact_size_mb=file_size_mb(str(copied)),
        weights_ref=weights_ref,
    )


def ensure_tensorrt_artifact(
    key: ArtifactKey,
    args: argparse.Namespace,
    weights_ref: str,
    get_base_model: Optional[Any],
) -> ArtifactInfo:
    if not device_supports_trt(args.device):
        raise RuntimeError("TensorRT requires a CUDA-capable device.")

    artifact_dir = artifact_base_dir(args, key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_file_path(args, key)
    if target is None:
        raise RuntimeError("Unsupported format key: %s" % key.format_key)

    onnx_key = ArtifactKey(
        model=key.model,
        imgsz=key.imgsz,
        format_key="onnx",
        half=key.half,
    )
    onnx_info = ensure_portable_artifact(onnx_key, args, weights_ref, get_base_model)
    onnx_path = Path(onnx_info.artifact_path)

    meta_path = artifact_metadata_path(args, key)
    meta = load_json(meta_path)
    if target.exists() and not args.rebuild_artifacts:
        if trt_artifact_metadata_matches(
            meta=meta,
            key=key,
            target=target,
            weights_ref=weights_ref,
            onnx_path=onnx_path,
            device=args.device,
        ):
            return ArtifactInfo(
                key=key,
                artifact_dir=artifact_dir,
                artifact_path=str(target),
                artifact_size_mb=file_size_mb(str(target)),
                weights_ref=weights_ref,
            )
        if meta is None and args.trust_existing_trt:
            return ArtifactInfo(
                key=key,
                artifact_dir=artifact_dir,
                artifact_path=str(target),
                artifact_size_mb=file_size_mb(str(target)),
                weights_ref=weights_ref,
            )

    built = build_tensorrt_engine_from_onnx(onnx_path, target, key, args)
    save_json(
        meta_path,
        build_trt_artifact_metadata(
            key=key,
            artifact_path=str(built),
            weights_ref=weights_ref,
            onnx_path=onnx_path,
            device=args.device,
        ),
    )
    return ArtifactInfo(
        key=key,
        artifact_dir=artifact_dir,
        artifact_path=str(built),
        artifact_size_mb=file_size_mb(str(built)),
        weights_ref=weights_ref,
    )


def ensure_artifact(
    key: ArtifactKey,
    args: argparse.Namespace,
    weights_ref: str,
    get_base_model: Optional[Any],
) -> ArtifactInfo:
    if key.format_key in PORTABLE_FORMATS:
        return ensure_portable_artifact(key, args, weights_ref, get_base_model)
    if key.format_key == "trt":
        return ensure_tensorrt_artifact(key, args, weights_ref, get_base_model)
    raise RuntimeError("Unsupported format key: %s" % key.format_key)


def artifact_or_weights_path(key: ArtifactKey, artifact_info: ArtifactInfo) -> str:
    return artifact_info.artifact_path


def warmup_model(
    model: Any,
    key: ArtifactKey,
    source_bundle: SourceBundle,
    args: argparse.Namespace,
) -> None:
    if not source_bundle.warmup_images:
        return

    # Important: Ultralytics treats a Python list of image paths as a batched source, and the
    # `batch` argument only applies to directory/video/.txt inputs. Our exported ONNX/TRT artifacts
    # are intentionally static batch=1, so warm up them one image at a time to preserve BCHW=(1,3,H,W).
    for image_path in source_bundle.warmup_images:
        for _ in model.predict(
            source=image_path,
            imgsz=int(key.imgsz),
            device=args.device,
            half=bool(key.half),
            conf=float(args.score_thr),
            max_det=int(args.max_det),
            batch=1,
            stream=True,
            save=False,
            verbose=bool(args.progress),
            workers=int(args.workers),
        ):
            pass


def run_predict_speed(
    model: Any,
    key: ArtifactKey,
    source_bundle: SourceBundle,
    args: argparse.Namespace,
) -> PredictPassResult:
    pre_total = 0.0
    inf_total = 0.0
    post_total = 0.0
    num_images = 0
    wall_start = time.perf_counter()

    iterator = model.predict(
        source=source_bundle.speed_source,
        imgsz=int(key.imgsz),
        device=args.device,
        half=bool(key.half),
        conf=float(args.score_thr),
        max_det=int(args.max_det),
        batch=1,
        stream=True,
        save=False,
        verbose=bool(args.progress),
        workers=int(args.workers),
    )

    result = None
    try:
        for result in iterator:
            speed = getattr(result, "speed", {}) or {}
            pre_total += float(speed.get("preprocess", 0.0) or 0.0)
            inf_total += float(speed.get("inference", 0.0) or 0.0)
            post_total += float(speed.get("postprocess", 0.0) or 0.0)
            num_images += 1
    finally:
        result = None
        iterator = None

    wall_time_s = time.perf_counter() - wall_start
    denom = float(max(1, num_images))
    return PredictPassResult(
        preprocess_ms=pre_total / denom,
        inference_ms=inf_total / denom,
        postprocess_ms=post_total / denom,
        wall_time_s=wall_time_s,
        num_images=num_images,
    )


def extract_metric_key(results_dict: Dict[str, Any]) -> str:
    for key in results_dict:
        if key.startswith("metrics/") and "mAP50-95" in key:
            return key
    for key in results_dict:
        if "mAP50-95" in key:
            return key
    return METRIC_KEY


def extract_val_result(metrics: Any) -> ValResult:
    speed = getattr(metrics, "speed", {}) or {}
    results_dict = getattr(metrics, "results_dict", {}) or {}

    ap50_95: Optional[float] = None
    if hasattr(metrics, "box") and hasattr(metrics.box, "map"):
        try:
            ap50_95 = float(metrics.box.map)
        except Exception:
            ap50_95 = None
    metric_key = extract_metric_key(results_dict) if results_dict else METRIC_KEY
    if ap50_95 is None:
        ap50_95 = parse_float(results_dict.get(metric_key))
        if ap50_95 is None:
            ap50_95 = float("nan")

    return ValResult(
        preprocess_ms=float(speed.get("preprocess", 0.0) or 0.0),
        inference_ms=float(speed.get("inference", 0.0) or 0.0),
        postprocess_ms=float(speed.get("postprocess", 0.0) or 0.0),
        wall_time_s=float("nan"),
        num_images=0,
        ap50_95=float(ap50_95),
        metric_key=metric_key,
    )


def run_validation(
    model: Any,
    key: ArtifactKey,
    args: argparse.Namespace,
) -> ValResult:
    wall_start = time.perf_counter()
    metrics = model.val(
        data=str(args.data),
        imgsz=int(key.imgsz),
        batch=1,
        device=args.device,
        half=bool(key.half),
        conf=float(args.score_thr),
        max_det=int(args.max_det),
        workers=int(args.workers),
        plots=False,
        save=False,
        verbose=bool(args.progress),
    )
    result = extract_val_result(metrics)
    result.wall_time_s = time.perf_counter() - wall_start
    return result


def eval_signature(
    key: ArtifactKey, args: argparse.Namespace, artifact_info: ArtifactInfo
) -> Dict[str, Any]:
    artifact_signature = None
    if artifact_info.artifact_path:
        artifact_signature = file_signature(artifact_info.artifact_path)

    data_signature: Dict[str, Any] = {"name": Path(args.data).name}
    data_file_sig = file_signature(args.data)
    if data_file_sig is not None:
        data_signature["file"] = data_file_sig

    return {
        "model": key.model,
        "imgsz": int(key.imgsz),
        "format_key": key.format_key,
        "half": bool(key.half),
        "data_signature": data_signature,
        "score_thr": float(args.score_thr),
        "max_det": int(args.max_det),
        "metric_key_default": METRIC_KEY,
        "artifact_signature": artifact_signature,
    }


def load_cached_metrics(
    key: ArtifactKey, args: argparse.Namespace, artifact_info: ArtifactInfo
) -> Optional[Dict[str, Any]]:
    path = metrics_json_path(args, key)
    if not path.exists() or not args.reuse_accuracy_cache:
        return None
    payload = load_json(path)
    if payload is None:
        return None
    expected = eval_signature(key, args, artifact_info)
    for name, value in expected.items():
        if payload.get(name) != value:
            return None
    return payload


def save_cached_metrics(
    key: ArtifactKey,
    args: argparse.Namespace,
    artifact_info: ArtifactInfo,
    result: ValResult,
) -> None:
    payload = eval_signature(key, args, artifact_info)
    payload.update(
        {
            "ap50_95": float(result.ap50_95),
            "metric_key": str(result.metric_key),
            "speed_preprocess_ms": float(result.preprocess_ms),
            "speed_inference_ms": float(result.inference_ms),
            "speed_postprocess_ms": float(result.postprocess_ms),
            "benchmark_wall_time_s": float(result.wall_time_s),
            "created_at_unix": float(time.time()),
        }
    )
    save_json(metrics_json_path(args, key), payload)


def spec_row_key(
    framework: str,
    key: ArtifactKey,
    weights_ref: str,
    args: argparse.Namespace,
    repeat: int,
) -> RowKey:
    return RowKey(
        framework=framework,
        model=key.model,
        weights=weights_ref,
        imgsz=int(key.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        repeat=int(repeat),
        half=int(key.half),
        precision=precision_name(key.half),
        format_name=FORMAT_TO_NAME[key.format_key],
        conf=round(float(args.score_thr), 6),
        max_det=int(args.max_det),
    )


def metric_lookup_key(
    key: ArtifactKey,
    weights_ref: str,
    args: argparse.Namespace,
) -> MetricLookupKey:
    return MetricLookupKey(
        model=key.model,
        weights=weights_ref,
        imgsz=int(key.imgsz),
        half=int(key.half),
        format_name=FORMAT_TO_NAME[key.format_key],
        conf=round(float(args.score_thr), 6),
        max_det=int(args.max_det),
    )


def build_error_row(
    key: ArtifactKey,
    args: argparse.Namespace,
    repeat: int,
    weights_ref: str,
    artifact_info: Optional[ArtifactInfo],
    error: str,
    metric_key: str = METRIC_KEY,
    model_stats: Optional[ModelStatsEntry] = None,
) -> Dict[str, Any]:
    expected_artifact = artifact_file_path(args, key)
    artifact_path = (
        artifact_info.artifact_path
        if artifact_info is not None
        else (str(expected_artifact) if expected_artifact is not None else weights_ref)
    )
    artifact_size_mb = (
        artifact_info.artifact_size_mb
        if artifact_info is not None
        else file_size_mb(artifact_path)
    )
    save_dir = (
        artifact_info.artifact_dir
        if artifact_info is not None
        else artifact_base_dir(args, key)
    )
    return {
        "framework": "ultralytics",
        "model": key.model,
        "weights": weights_ref,
        "imgsz": int(key.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "repeat": int(repeat),
        "half": int(key.half),
        "precision": precision_name(key.half),
        "format_name": FORMAT_TO_NAME[key.format_key],
        "format_arg": FORMAT_TO_EXPORT[key.format_key],
        "backend": FORMAT_TO_NAME[key.format_key].lower(),
        "runtime_provider": "",
        "benchmark_impl": BENCHMARK_IMPL,
        "input_geometry": "ultralytics_predict_default",
        "resize_mode": "ultralytics_default",
        "conf": float(args.score_thr),
        "max_det": int(args.max_det),
        "params": (
            model_stats.params
            if model_stats is not None and model_stats.params is not None
            else ""
        ),
        "gflops": (
            model_stats.gflops
            if model_stats is not None and model_stats.gflops is not None
            else ""
        ),
        "speed_preprocess_ms": "",
        "speed_inference_ms": "",
        "speed_postprocess_ms": "",
        "ap50_95": "",
        "fps": "",
        "status": "error",
        "error": error,
        "artifact_path": artifact_path,
        "artifact_size_mb": artifact_size_mb if artifact_size_mb is not None else "",
        "save_dir": str(save_dir),
        "run_name": run_name_for(key, repeat),
        "metric_key": metric_key,
        "benchmark_wall_time_s": "",
    }


def build_result_row(
    key: ArtifactKey,
    args: argparse.Namespace,
    repeat: int,
    artifact_info: ArtifactInfo,
    predict_result: PredictPassResult,
    ap50_95: float,
    metric_key: str,
    model_stats: Optional[ModelStatsEntry],
) -> Dict[str, Any]:
    fps = (
        None
        if predict_result.inference_ms <= 0
        else (1000.0 / float(predict_result.inference_ms))
    )
    return {
        "framework": "ultralytics",
        "model": key.model,
        "weights": artifact_info.weights_ref,
        "imgsz": int(key.imgsz),
        "batch": int(args.batch),
        "device": str(args.device),
        "repeat": int(repeat),
        "half": int(key.half),
        "precision": precision_name(key.half),
        "format_name": FORMAT_TO_NAME[key.format_key],
        "format_arg": FORMAT_TO_EXPORT[key.format_key],
        "backend": FORMAT_TO_NAME[key.format_key].lower(),
        "runtime_provider": "ultralytics_autobackend",
        "benchmark_impl": BENCHMARK_IMPL,
        "input_geometry": "ultralytics_predict_default",
        "resize_mode": "ultralytics_default",
        "conf": float(args.score_thr),
        "max_det": int(args.max_det),
        "params": (
            model_stats.params
            if model_stats is not None and model_stats.params is not None
            else ""
        ),
        "gflops": (
            model_stats.gflops
            if model_stats is not None and model_stats.gflops is not None
            else ""
        ),
        "speed_preprocess_ms": float(predict_result.preprocess_ms),
        "speed_inference_ms": float(predict_result.inference_ms),
        "speed_postprocess_ms": float(predict_result.postprocess_ms),
        "ap50_95": float(ap50_95),
        "fps": fps,
        "status": "success",
        "error": "",
        "artifact_path": artifact_info.artifact_path,
        "artifact_size_mb": (
            artifact_info.artifact_size_mb
            if artifact_info.artifact_size_mb is not None
            else ""
        ),
        "save_dir": str(artifact_info.artifact_dir),
        "run_name": run_name_for(key, repeat),
        "metric_key": metric_key,
        "benchmark_wall_time_s": float(predict_result.wall_time_s),
    }


def all_repeats_already_done(
    experiment: ExperimentSpec,
    args: argparse.Namespace,
    existing_success: Set[RowKey],
) -> bool:
    if not args.resume:
        return False
    for repeat in range(1, int(args.repeats) + 1):
        if (
            spec_row_key(
                framework="ultralytics",
                key=experiment.key,
                weights_ref=experiment.weights_ref,
                args=args,
                repeat=repeat,
            )
            not in existing_success
        ):
            return False
    return True


def plan_experiments(
    args: argparse.Namespace,
) -> Tuple[List[ExperimentSpec], List[str]]:
    experiments: List[ExperimentSpec] = []
    skipped: List[str] = []
    precisions = [False]
    if args.include_half:
        precisions.append(True)

    for model_name in args.models:
        if is_excluded_model(model_name):
            skipped.append("Excluded model family skipped: %s" % model_name)
            continue
        weights_ref = resolve_model_reference(model_name, args.weights_dir)
        for imgsz in args.imgsz:
            for half in precisions:
                if half and not device_supports_half(args.device):
                    skipped.append(
                        "FP16 skipped for %s imgsz=%s on device=%s"
                        % (model_name, imgsz, args.device)
                    )
                    continue
                for format_key in args.formats:
                    if format_key == "trt" and not device_supports_trt(args.device):
                        skipped.append(
                            "TensorRT skipped for %s imgsz=%s on device=%s"
                            % (model_name, imgsz, args.device)
                        )
                        continue
                    experiments.append(
                        ExperimentSpec(
                            key=ArtifactKey(
                                model=str(model_name),
                                imgsz=int(imgsz),
                                format_key=str(format_key),
                                half=bool(half),
                            ),
                            weights_ref=weights_ref,
                        )
                    )
    experiments.sort(
        key=lambda item: (
            item.key.model,
            int(item.key.imgsz),
            int(item.key.half),
            FORMAT_ORDER[item.key.format_key],
        )
    )
    return experiments, skipped


def print_prepared_or_error(
    key: ArtifactKey,
    artifact_info: Optional[ArtifactInfo],
    error: Optional[str],
) -> None:
    if error is None and artifact_info is not None:
        print(
            "[PREPARED] ultralytics %s imgsz=%s fmt=%s prec=%s -> %s"
            % (
                key.model,
                key.imgsz,
                key.format_key,
                precision_name(key.half),
                artifact_info.artifact_path or str(artifact_info.artifact_dir),
            )
        )
        return
    print(
        "[ERROR] ultralytics %s imgsz=%s fmt=%s prec=%s -> %s"
        % (
            key.model,
            key.imgsz,
            key.format_key,
            precision_name(key.half),
            error or "unknown error",
        )
    )


def print_row_summary(row: Dict[str, Any]) -> None:
    print(
        "[ultralytics] {model:>12} imgsz={imgsz:>4} fmt={fmt:>11} prec={prec:>4} run={run:>2} | "
        "status={status:>8} inf={inf}ms AP50-95={ap}".format(
            model=row["model"],
            imgsz=row["imgsz"],
            fmt=row["format_name"],
            prec=row["precision"],
            run=row["repeat"],
            status=row["status"],
            inf=printable(parse_float(row.get("speed_inference_ms"))),
            ap=printable(parse_float(row.get("ap50_95"))),
        )
    )
    if row.get("error"):
        print("  -> %s" % row["error"])


def prepare_artifacts(
    args: argparse.Namespace,
    YOLO: Any,
    experiments: Sequence[ExperimentSpec],
    existing_model_stats: Dict[ModelStatsKey, ModelStatsEntry],
) -> Tuple[
    Dict[ArtifactKey, ArtifactInfo],
    Dict[ArtifactKey, str],
    Dict[ModelStatsKey, ModelStatsEntry],
]:
    prepared: Dict[ArtifactKey, ArtifactInfo] = {}
    errors: Dict[ArtifactKey, str] = {}
    model_stats: Dict[ModelStatsKey, ModelStatsEntry] = dict(existing_model_stats)
    by_model: Dict[str, List[ExperimentSpec]] = {}
    for experiment in experiments:
        by_model.setdefault(experiment.key.model, []).append(experiment)

    for model_name in sorted(by_model):
        model_experiments = by_model[model_name]
        weights_ref = model_experiments[0].weights_ref
        base_model: Any = None

        def get_base_model() -> Any:
            nonlocal base_model
            if base_model is None:
                base_model = YOLO(weights_ref)
            return base_model

        unique_keys: Dict[ArtifactKey, ExperimentSpec] = {}
        for experiment in model_experiments:
            unique_keys[experiment.key] = experiment

        model_stats.update(
            prepare_model_stats_for_model(
                args=args,
                model_name=model_name,
                weights_ref=weights_ref,
                imgsz_values=[experiment.key.imgsz for experiment in model_experiments],
                get_base_model=get_base_model,
                existing_model_stats=model_stats,
            )
        )

        for key in sorted(
            unique_keys,
            key=lambda item: (
                item.model,
                int(item.imgsz),
                int(item.half),
                FORMAT_ORDER[item.format_key],
            ),
        ):
            try:
                artifact_info = ensure_artifact(
                    key=key,
                    args=args,
                    weights_ref=weights_ref,
                    get_base_model=get_base_model if args.mode == "full" else None,
                )
                prepared[key] = artifact_info
                if args.prepare_only:
                    print_prepared_or_error(key, artifact_info, None)
            except Exception as exc:
                errors[key] = str(exc)
                if args.prepare_only:
                    print_prepared_or_error(key, None, str(exc))
            finally:
                gc_cuda_barrier(args.device)

        release_yolo_runtime(base_model, args.device)
        base_model = None
    return prepared, errors, model_stats


def benchmark_round_robin(
    args: argparse.Namespace,
    YOLO: Any,
    experiments: Sequence[ExperimentSpec],
    prepared: Dict[ArtifactKey, ArtifactInfo],
    prepare_errors: Dict[ArtifactKey, str],
    model_stats: Dict[ModelStatsKey, ModelStatsEntry],
    source_bundle: SourceBundle,
    existing_success: Set[RowKey],
    existing_metric_rows: Dict[MetricLookupKey, MetricCacheEntry],
) -> int:
    metric_cache: Dict[MetricLookupKey, MetricCacheEntry] = dict(existing_metric_rows)
    rows_written = 0

    active_experiments: List[ExperimentSpec] = []
    for experiment in experiments:
        if all_repeats_already_done(experiment, args, existing_success):
            print(
                "[SKIP] ultralytics %s imgsz=%s fmt=%s prec=%s | all repeats already succeeded in %s"
                % (
                    experiment.key.model,
                    experiment.key.imgsz,
                    experiment.key.format_key,
                    precision_name(experiment.key.half),
                    args.out_csv,
                )
            )
            continue
        active_experiments.append(experiment)

    if not active_experiments:
        return 0

    for repeat in range(1, int(args.repeats) + 1):
        print("\n[REPEAT %d/%d]" % (repeat, int(args.repeats)))
        for experiment in active_experiments:
            key = experiment.key
            weights_ref = experiment.weights_ref
            stats_entry = model_stats.get(model_stats_lookup_key(key, weights_ref))
            row_key = spec_row_key(
                framework="ultralytics",
                key=key,
                weights_ref=weights_ref,
                args=args,
                repeat=repeat,
            )
            if args.resume and row_key in existing_success:
                continue

            artifact_info = prepared.get(key)
            prep_error = prepare_errors.get(key)
            if prep_error is not None or artifact_info is None:
                row = build_error_row(
                    key=key,
                    args=args,
                    repeat=repeat,
                    weights_ref=weights_ref,
                    artifact_info=artifact_info,
                    error=prep_error or "Artifact preparation failed.",
                    model_stats=stats_entry,
                )
                write_csv([row], args.out_csv)
                print_row_summary(row)
                rows_written += 1
                continue

            metric_key_lookup = metric_lookup_key(key, weights_ref, args)
            try:
                runtime_path = artifact_or_weights_path(key, artifact_info)

                if args.eval_policy == "every-repeat":
                    val_result = run_validation_isolated(YOLO, runtime_path, key, args)
                    metric_entry = MetricCacheEntry(
                        ap50_95=float(val_result.ap50_95),
                        metric_key=str(val_result.metric_key),
                    )
                    metric_cache[metric_key_lookup] = metric_entry
                    predict_result = PredictPassResult(
                        preprocess_ms=val_result.preprocess_ms,
                        inference_ms=val_result.inference_ms,
                        postprocess_ms=val_result.postprocess_ms,
                        wall_time_s=val_result.wall_time_s,
                        num_images=val_result.num_images,
                    )
                else:
                    metric_entry = metric_cache.get(metric_key_lookup)
                    if metric_entry is None:
                        cached_metrics = load_cached_metrics(key, args, artifact_info)
                        if cached_metrics is not None:
                            metric_entry = MetricCacheEntry(
                                ap50_95=float(cached_metrics["ap50_95"]),
                                metric_key=safe_str(cached_metrics.get("metric_key"))
                                or METRIC_KEY,
                            )
                        else:
                            val_result = run_validation_isolated(
                                YOLO, runtime_path, key, args
                            )
                            save_cached_metrics(key, args, artifact_info, val_result)
                            metric_entry = MetricCacheEntry(
                                ap50_95=float(val_result.ap50_95),
                                metric_key=str(val_result.metric_key),
                            )
                        metric_cache[metric_key_lookup] = metric_entry
                    predict_result = run_predict_speed_isolated(
                        YOLO=YOLO,
                        runtime_path=runtime_path,
                        key=key,
                        source_bundle=source_bundle,
                        args=args,
                    )

                row = build_result_row(
                    key=key,
                    args=args,
                    repeat=repeat,
                    artifact_info=artifact_info,
                    predict_result=predict_result,
                    ap50_95=float(metric_entry.ap50_95),
                    metric_key=str(metric_entry.metric_key),
                    model_stats=stats_entry,
                )
            except Exception as exc:
                row = build_error_row(
                    key=key,
                    args=args,
                    repeat=repeat,
                    weights_ref=weights_ref,
                    artifact_info=artifact_info,
                    error=str(exc),
                    metric_key=(
                        metric_cache.get(metric_key_lookup).metric_key  # type: ignore
                        if metric_key_lookup in metric_cache
                        else METRIC_KEY
                    ),
                    model_stats=stats_entry,
                )
            finally:
                gc_cuda_barrier(args.device)

            write_csv([row], args.out_csv)
            print_row_summary(row)
            rows_written += 1
            if row.get("status") == "success":
                existing_success.add(row_key)

    return rows_written


def main() -> None:
    args = parse_args()
    if int(args.batch) != 1:
        raise SystemExit("This script keeps batch=1 to match the benchmark intent.")

    experiments, skipped = plan_experiments(args)
    if skipped:
        seen: Set[str] = set()
        for message in skipped:
            if message in seen:
                continue
            print("[INFO] %s" % message)
            seen.add(message)

    if not experiments:
        print("No experiments were scheduled.")
        return

    source_bundle = None
    if not args.prepare_only:
        source_bundle = build_source_bundle(args.data, args.warmup_images)

    YOLO = ensure_yolo_import()
    existing_success = (
        load_existing_success_keys(args.out_csv) if args.resume else set()
    )
    existing_metric_rows = (
        load_existing_metric_rows(args.out_csv) if args.resume else {}
    )
    existing_model_stats_rows = (
        load_existing_model_stats_rows(args.out_csv) if args.resume else {}
    )

    prepared, prepare_errors, model_stats = prepare_artifacts(
        args,
        YOLO,
        experiments,
        existing_model_stats_rows,
    )
    gc_cuda_barrier(args.device)

    if args.prepare_only:
        print("\nPrepared artifacts under: %s" % args.artifact_root)
        return

    assert source_bundle is not None
    rows_written = benchmark_round_robin(
        args=args,
        YOLO=YOLO,
        experiments=experiments,
        prepared=prepared,
        prepare_errors=prepare_errors,
        model_stats=model_stats,
        source_bundle=source_bundle,
        existing_success=existing_success,
        existing_metric_rows=existing_metric_rows,
    )

    print("\nSaved %d rows to: %s" % (rows_written, args.out_csv))


if __name__ == "__main__":
    main()
