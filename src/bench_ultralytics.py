#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_MODELS = [
    "yolo26n",
    # "yolo26s",
    # "yolo26m",
    # "yolo26l",
    # "yolo26x",
    # "yolo11n",
    # "yolo11s",
    # "yolo11m",
    # "yolo11l",
    # "yolo11x",
    # "yolov5nu",
    # "yolov5mu",
    # "yolov5xu",
]
DEFAULT_SIZES = [384, 512, 640, 768]
DEFAULT_FORMATS = ["pytorch", "torchscript", "onnx", "trt"]
FORMAT_TO_ARG = {
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


@dataclass(frozen=True)
class RunSpec:
    model: str
    imgsz: int
    format_key: str
    half: bool
    repeat: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ultralytics detection models using current benchmark mode."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model names to benchmark. RT-DETR and YOLO-NAS are intentionally excluded by default.",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="One or more square benchmark sizes.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=sorted(FORMAT_TO_ARG.keys()),
        help="Benchmark formats to run.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of independent benchmark repeats to run per model/size/format/precision.",
    )
    parser.add_argument(
        "--include-half",
        dest="include_half",
        action="store_true",
        default=True,
        help="Include FP16 benchmark rows in addition to FP32 rows (default: enabled).",
    )
    parser.add_argument(
        "--no-include-half",
        dest="include_half",
        action="store_false",
        help="Disable FP16 benchmark rows.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/coco.yaml",
        help="Dataset YAML for benchmark validation.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("weights"),
        help="Directory containing local .pt weights. If a file is missing, the script falls back to '<model>.pt'.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Benchmark batch size. Kept at 1 for parity with Ultralytics benchmark mode.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help='Device passed to Ultralytics, e.g. 0, "cuda:0", or "cpu".',
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/val_bench",
        help="Unused by Ultralytics benchmark() itself, but retained in the CSV for continuity.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("val_jetson_test.csv"),
        help="CSV file to append results to.",
    )
    parser.add_argument(
        "--exist-ok",
        dest="exist_ok",
        action="store_true",
        default=True,
        help="Retained for interface continuity.",
    )
    parser.add_argument(
        "--no-exist-ok",
        dest="exist_ok",
        action="store_false",
        help="Retained for interface continuity.",
    )
    return parser.parse_args()


def is_excluded_model(name: str) -> bool:
    low = name.lower()
    return "rtdetr" in low or "rt-detr" in low or "yolo_nas" in low or "yolo-nas" in low


def resolve_model_reference(model_name: str, weights_dir: Path) -> str:
    candidate = (weights_dir / (model_name + ".pt")).expanduser()
    if candidate.exists():
        return str(candidate)
    return model_name + ".pt"


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


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def metric_key_from_record(record: Dict[str, Any]) -> str:
    for key in record:
        if key.startswith("metrics/") and "mAP50-95" in key:
            return key
    for key in record:
        if "mAP50-95" in key:
            return key
    return "metrics/mAP50-95(B)"


def normalize_benchmark_records(df: Any) -> List[Dict[str, Any]]:
    if df is None:
        return []
    if hasattr(df, "to_dicts"):
        try:
            records = df.to_dicts()
            if isinstance(records, list):
                return list(records)
        except Exception:
            pass
    if hasattr(df, "iter_rows"):
        try:
            return list(df.iter_rows(named=True))
        except Exception:
            pass
    if hasattr(df, "to_dict"):
        try:
            records = df.to_dict("records")
            if isinstance(records, list):
                return list(records)
        except Exception:
            pass
        try:
            raw = df.to_dict()
            if isinstance(raw, list):
                return list(raw)
        except Exception:
            pass
    if isinstance(df, list):
        return list(df)
    raise TypeError("Unsupported benchmark result type: %r" % (type(df),))


def infer_status(raw_status: str) -> str:
    raw_status = (raw_status or "").strip()
    if raw_status == "✅":
        return "success"
    if raw_status == "❎":
        return "validation_failed"
    if raw_status == "❌":
        return "export_failed"
    if raw_status == "❌️":
        return "export_failed"
    return raw_status or "unknown"


def infer_artifact_path(base_pt_path: Optional[Path], format_key: str) -> str:
    if base_pt_path is None:
        return ""
    if format_key == "pytorch":
        return str(base_pt_path)
    suffix = {
        "torchscript": ".torchscript",
        "onnx": ".onnx",
        "trt": ".engine",
    }.get(format_key)
    if not suffix:
        return ""
    return str(base_pt_path.with_suffix(suffix))


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


def ensure_csv_schema(out_csv: Path) -> None:
    out_csv = out_csv.expanduser().resolve()
    if not out_csv.exists():
        return
    try:
        with out_csv.open("r", newline="") as handle:
            first_line = handle.readline().strip("\n\r")
    except Exception:
        return
    current_header = first_line.split(",") if first_line else []
    if current_header == FIELDNAMES:
        return
    backup = out_csv.with_name(
        out_csv.stem + ".legacy_" + str(int(time.time())) + out_csv.suffix
    )
    out_csv.rename(backup)
    print(
        "[INFO] Existing CSV header differs from the new schema. Moved old file to: %s"
        % backup
    )


def write_csv(rows: Iterable[Dict[str, Any]], out_csv: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    out_csv = out_csv.expanduser().resolve()
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


def build_error_row(
    spec: RunSpec, args: argparse.Namespace, weights_ref: str, error: str
) -> Dict[str, Any]:
    format_arg = FORMAT_TO_ARG[spec.format_key]
    return {
        "framework": "ultralytics",
        "model": spec.model,
        "weights": weights_ref,
        "imgsz": spec.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "repeat": spec.repeat,
        "half": int(spec.half),
        "precision": "fp16" if spec.half else "fp32",
        "format_name": FORMAT_TO_NAME[spec.format_key],
        "format_arg": format_arg,
        "backend": FORMAT_TO_NAME[spec.format_key].lower(),
        "runtime_provider": "",
        "benchmark_impl": "ultralytics.Model.benchmark",
        "input_geometry": "ultralytics benchmark default (batch=1)",
        "resize_mode": "ultralytics_default",
        "conf": 0.001,
        "max_det": 100,
        "speed_preprocess_ms": "",
        "speed_inference_ms": "",
        "speed_postprocess_ms": "",
        "ap50_95": "",
        "fps": "",
        "status": "error",
        "error": error,
        "artifact_path": "",
        "artifact_size_mb": "",
        "save_dir": str(args.project),
        "run_name": "{model}_img{imgsz}_{fmt}_{prec}_r{repeat}".format(
            model=spec.model,
            imgsz=spec.imgsz,
            fmt=spec.format_key,
            prec="fp16" if spec.half else "fp32",
            repeat=spec.repeat,
        ),
        "metric_key": "metrics/mAP50-95(B)",
        "benchmark_wall_time_s": "",
    }


def run_one(spec: RunSpec, args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.batch) != 1:
        return build_error_row(
            spec,
            args,
            resolve_model_reference(spec.model, args.weights_dir),
            "Ultralytics benchmark mode validates with batch=1 only for this script.",
        )

    weights_ref = resolve_model_reference(spec.model, args.weights_dir)
    format_arg = FORMAT_TO_ARG[spec.format_key]
    precision = "fp16" if spec.half else "fp32"

    try:
        from ultralytics import YOLO
    except Exception as exc:
        return build_error_row(
            spec, args, weights_ref, "Failed to import ultralytics: %s" % exc
        )

    try:
        model = YOLO(weights_ref)
    except Exception as exc:
        return build_error_row(
            spec,
            args,
            weights_ref,
            "Failed to load model '%s': %s" % (weights_ref, exc),
        )

    base_pt_path = None  # type: Optional[Path]
    for attr in ("ckpt_path", "pt_path"):
        value = getattr(model, attr, None)
        if value:
            try:
                base_pt_path = Path(str(value)).expanduser().resolve()
                break
            except Exception:
                pass
    if base_pt_path is None:
        try:
            if weights_ref.endswith(".pt"):
                base_pt_path = Path(weights_ref).expanduser().resolve()
        except Exception:
            base_pt_path = None

    wall_start = time.perf_counter()
    benchmark_impl = "ultralytics.Model.benchmark"
    try:
        if spec.half and format_arg == "-":
            # Current Model.benchmark() filters export kwargs by format, which drops half=True for '-' (PyTorch).
            # Use the underlying benchmark() directly here so the FP16 PyTorch row is actually benchmarked.
            from ultralytics.utils.benchmarks import benchmark as ul_benchmark

            benchmark_impl = "ultralytics.utils.benchmarks.benchmark"
            df = ul_benchmark(
                model=model,  # type: ignore
                data=args.data,
                imgsz=int(spec.imgsz),
                device=args.device,
                half=True,
                verbose=False,
                format=format_arg,
            )
        else:
            df = model.benchmark(
                data=args.data,
                imgsz=int(spec.imgsz),
                device=args.device,
                half=bool(spec.half),
                verbose=False,
                format=format_arg,
            )
        wall_time_s = time.perf_counter() - wall_start
    except Exception as exc:
        return build_error_row(
            spec, args, weights_ref, "Benchmark call failed: %s" % exc
        )

    try:
        records = normalize_benchmark_records(df)
    except Exception as exc:
        return build_error_row(
            spec, args, weights_ref, "Failed to parse benchmark output: %s" % exc
        )

    target_record = None  # type: Optional[Dict[str, Any]]
    for record in records:
        if safe_str(record.get("Format")) == FORMAT_TO_NAME[spec.format_key]:
            target_record = record
            break
    if target_record is None and records:
        target_record = records[0]
    if target_record is None:
        return build_error_row(
            spec, args, weights_ref, "Ultralytics benchmark returned no rows."
        )

    metric_key = metric_key_from_record(target_record)
    raw_status = safe_str(target_record.get("Status❔"))
    status = infer_status(raw_status)
    artifact_path = infer_artifact_path(base_pt_path, spec.format_key)
    parsed_size = parse_float(target_record.get("Size (MB)"))
    inferred_size = file_size_mb(artifact_path)
    artifact_size = parsed_size if parsed_size is not None else inferred_size
    save_dir = ""
    if artifact_path:
        try:
            save_dir = str(Path(artifact_path).expanduser().resolve().parent)
        except Exception:
            save_dir = ""

    error_text = ""
    if status != "success":
        error_text = "Ultralytics benchmark reported status %s for %s." % (
            raw_status or "<empty>",
            FORMAT_TO_NAME[spec.format_key],
        )

    row = {
        "framework": "ultralytics",
        "model": spec.model,
        "weights": weights_ref,
        "imgsz": spec.imgsz,
        "batch": 1,
        "device": str(args.device),
        "repeat": spec.repeat,
        "half": int(spec.half),
        "precision": precision,
        "format_name": safe_str(target_record.get("Format"))
        or FORMAT_TO_NAME[spec.format_key],
        "format_arg": format_arg,
        "backend": FORMAT_TO_NAME[spec.format_key].lower(),
        "runtime_provider": "ultralytics_autobackend",
        "benchmark_impl": benchmark_impl,
        "input_geometry": "ultralytics benchmark default (batch=1)",
        "resize_mode": "ultralytics_default",
        "conf": 0.001,
        "max_det": 100,
        "speed_preprocess_ms": "",
        "speed_inference_ms": parse_float(target_record.get("Inference time (ms/im)")),
        "speed_postprocess_ms": "",
        "ap50_95": parse_float(target_record.get(metric_key)),
        "fps": parse_float(target_record.get("FPS")),
        "status": status,
        "error": error_text,
        "artifact_path": artifact_path,
        "artifact_size_mb": artifact_size,
        "save_dir": save_dir,
        "run_name": "{model}_img{imgsz}_{fmt}_{prec}_r{repeat}".format(
            model=spec.model,
            imgsz=spec.imgsz,
            fmt=spec.format_key,
            prec=precision,
            repeat=spec.repeat,
        ),
        "metric_key": metric_key,
        "benchmark_wall_time_s": wall_time_s,
    }
    return row


def printable(value: Any) -> str:
    if value is None:
        return "nan"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return "%.4f" % value
    return str(value)


def main() -> None:
    args = parse_args()

    precisions = [False]
    if args.include_half:
        precisions.append(True)

    rows_written = 0
    for model_name in args.models:
        if is_excluded_model(model_name):
            print("[INFO] Skipping excluded model family: %s" % model_name)
            continue
        for imgsz in args.imgsz:
            for format_key in args.formats:
                for half in precisions:
                    for repeat in range(1, int(args.repeats) + 1):
                        spec = RunSpec(
                            model=str(model_name),
                            imgsz=int(imgsz),
                            format_key=str(format_key),
                            half=bool(half),
                            repeat=int(repeat),
                        )
                        row = run_one(spec, args)
                        print(
                            "[{framework}] {model:>12} imgsz={imgsz:>4} fmt={fmt:>11} prec={prec:>4} run={run:>2} | "
                            "status={status:>16} inf={inf}ms AP50-95={ap}".format(
                                framework=row["framework"],
                                model=row["model"],
                                imgsz=row["imgsz"],
                                fmt=row["format_name"],
                                prec=row["precision"],
                                run=row["repeat"],
                                status=row["status"],
                                inf=printable(row.get("speed_inference_ms")),
                                ap=printable(row.get("ap50_95")),
                            )
                        )
                        if row.get("error"):
                            print("  -> %s" % row["error"])
                        write_csv([row], args.out_csv)
                        rows_written += 1

    print(
        "\nSaved %d rows to: %s" % (rows_written, args.out_csv.expanduser().resolve())
    )


if __name__ == "__main__":
    main()
